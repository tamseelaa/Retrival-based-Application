import json
import logging
import os
from typing import Any, Dict, List, Optional
import psycopg
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)


class PgVectorStorage:
    """PostgreSQL pgvector storage for RAG document embeddings.

    Supports connection via DATABASE_URL or individual PG* / POSTGRES_* environment variables.
    Works seamlessly with local PostgreSQL (with pgvector) and AWS RDS / Aurora PostgreSQL.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "documents",
        dim: Optional[int] = None,
    ):
        self.connection_string = connection_string or self._get_connection_string()
        self.table_name = self._sanitize_identifier(table_name)
        if dim is not None:
            self.dim = dim
        else:
            try:
                from data_loader import EMBED_DIM
                self.dim = EMBED_DIM
            except Exception:
                self.dim = int(os.getenv("EMBED_DIM", "768"))
        self._init_db()

    def _get_connection_string(self) -> str:
        """Resolve PostgreSQL connection string from environment variables."""
        if os.getenv("DATABASE_URL"):
            return os.getenv("DATABASE_URL")

        host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres")
        dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "rag_db")

        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        """Ensure table name contains only valid alphanumeric characters and underscores."""
        clean_name = "".join(c for c in name if c.isalnum() or c == "_")
        return clean_name or "documents"

    def _get_connection(self) -> psycopg.Connection:
        """Create and configure a psycopg connection with pgvector registered."""
        conn = psycopg.connect(self.connection_string, autocommit=True)
        register_vector(conn)
        return conn

    def _init_db(self) -> None:
        """Initialize pgvector extension, table schema, and HNSW index."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # Re-register vector type in case extension was just created
                    register_vector(conn)

                    # Create documents table
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id TEXT PRIMARY KEY,
                            text TEXT NOT NULL,
                            source TEXT,
                            metadata JSONB,
                            embedding vector({self.dim}),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )

                    # Create vector similarity search index
                    if self.dim <= 2000:
                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx
                            ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
                            """
                        )
                    else:
                        try:
                            cur.execute(
                                f"""
                                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx
                                ON {self.table_name} USING hnsw ((embedding::halfvec({self.dim})) halfvec_cosine_ops);
                                """
                            )
                        except Exception as index_err:
                            logger.info(
                                f"Exact KNN scan will be used for dim={self.dim} (>2000 dimensions): {index_err}"
                            )
        except Exception as e:
            logger.warning(
                f"Database initialization warning (will retry on query if transient): {e}"
            )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """Batch upsert document chunks, embeddings, and metadata into PostgreSQL."""
        if not ids:
            return

        rows = []
        for i in range(len(ids)):
            doc_id = str(ids[i])
            vec = vectors[i]
            payload = payloads[i] if i < len(payloads) else {}

            text = payload.get("text", "")
            source = payload.get("source", "")
            metadata_json = json.dumps(payload)
            vec_str = f"[{','.join(map(str, vec))}]"

            rows.append((doc_id, text, source, metadata_json, vec_str))

        upsert_query = f"""
            INSERT INTO {self.table_name} (id, text, source, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                source = EXCLUDED.source,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(upsert_query, rows)

    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, Any]:
        """Perform cosine similarity search on embeddings."""
        search_query = f"""
            SELECT text, source
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        contexts: List[str] = []
        sources = set()
        vec_str = f"[{','.join(map(str, query_vector))}]"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(search_query, (vec_str, top_k))
                rows = cur.fetchall()

                for row in rows:
                    text = row[0] or ""
                    source = row[1] or ""
                    if text:
                        contexts.append(text)
                    if source:
                        sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}


# Alias for backward compatibility if needed
QdrantStorage = PgVectorStorage
VectorStorage = PgVectorStorage