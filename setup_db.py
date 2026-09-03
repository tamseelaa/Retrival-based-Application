import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    print("=" * 60)
    print("🚀 PostgreSQL + pgvector Database Setup & Verification")
    print("=" * 60)

    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        print("❌ Error: Missing required packages ('psycopg' or 'pgvector').")
        print("Run: pip install psycopg[binary] pgvector\n")
        sys.exit(1)

    # Determine connection string
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        host = os.getenv("PGHOST") or os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("PGPORT") or os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres")
        dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB", "rag_db")
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    # Mask password for display
    display_url = db_url
    if "@" in display_url and ":" in display_url.split("@")[0]:
        prefix, rest = display_url.split("://", 1)
        user_pass, host_db = rest.split("@", 1)
        user = user_pass.split(":")[0]
        display_url = f"{prefix}://{user}:****@{host_db}"

    try:
        from data_loader import EMBED_DIM, EMBED_MODEL
        dim = EMBED_DIM
        print(f"Embedding Model: {EMBED_MODEL} (Dimensions: {dim})")
    except Exception:
        dim = int(os.getenv("EMBED_DIM", "768"))

    table_name = "documents"

    try:
        conn = psycopg.connect(db_url, autocommit=True)
    except Exception as e:
        print("\n❌ Failed to connect to PostgreSQL!")
        print(f"Error: {e}\n")
        print("💡 Checklist:")
        print("1. If using Docker: Ensure container is running -> 'docker compose up -d'")
        print("2. If using Homebrew: Ensure Postgres is started -> 'brew services start postgresql@16'")
        print("3. If using Cloud/RDS: Check DATABASE_URL in your .env file")
        sys.exit(1)

    with conn:
        with conn.cursor() as cur:
            # 1. Enable pgvector extension
            print("\n1. Enabling 'vector' extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)
            print("   ✅ pgvector extension enabled.")

            # Check if table exists with mismatched dimension
            cur.execute(f"""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = (to_regclass('{table_name}')) AND attname = 'embedding';
            """)
            dim_row = cur.fetchone()
            if dim_row and dim_row[0] > 0 and dim_row[0] != dim:
                print(f"   ℹ️ Dimension change detected ({dim_row[0]} -> {dim}). Rebuilding '{table_name}' table...")
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

            # 2. Create documents table
            print(f"\n2. Creating '{table_name}' table (vector dimension = {dim})...")
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                metadata JSONB,
                embedding vector({dim}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            cur.execute(create_table_sql)
            print(f"   ✅ Table '{table_name}' created / verified.")

            # 3. Create Cosine Index (HNSW)
            print(f"\n3. Creating vector search index...")
            if dim <= 2000:
                create_index_sql = f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx
                ON {table_name} USING hnsw (embedding vector_cosine_ops);
                """
                cur.execute(create_index_sql)
                print(f"   ✅ Standard HNSW cosine index created ({dim} dims).")
            else:
                # pgvector supports up to 4096 dimensions for HNSW using halfvec
                try:
                    create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx
                    ON {table_name} USING hnsw ((embedding::halfvec({dim})) halfvec_cosine_ops);
                    """
                    cur.execute(create_index_sql)
                    print(f"   ✅ HNSW halfvec cosine index created ({dim} dims).")
                except Exception as ex:
                    print(f"   ℹ️ Note: High-dimension vector ({dim} dims). Using exact KNN sequential scan: {ex}")

            # 4. Display Table Schema
            print(f"\n4. Verifying Table Columns:")
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            columns = cur.fetchall()
            print(f"   {'-'*50}")
            print(f"   {'Column':<15} {'Type':<20} {'Nullable':<10}")
            print(f"   {'-'*50}")
            for col, dtype, null in columns:
                print(f"   {col:<15} {dtype:<20} {null:<10}")
            print(f"   {'-'*50}")

            # 5. Perform a test write & search
            print("\n5. Testing Vector Insert & Search...")
            test_id = "__test_setup_probe__"
            dummy_vec = [0.01] * dim
            cur.execute(
                f"""
                INSERT INTO {table_name} (id, text, source, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text;
                """,
                (test_id, "Probe vector test", "setup_probe", '{"probe": true}', dummy_vec)
            )

            # Search probe
            cur.execute(
                f"""
                SELECT id, text, (embedding <=> %s::vector) AS dist
                FROM {table_name}
                WHERE id = %s;
                """,
                (dummy_vec, test_id)
            )
            res = cur.fetchone()
            # Clean up test probe
            cur.execute(f"DELETE FROM {table_name} WHERE id = %s;", (test_id,))

            if res:
                print("   ✅ Test vector insert and search succeeded!")

    print("\n" + "=" * 60)
    print("🎉 Database setup complete! Your PostgreSQL pgvector DB is 100% ready.")
    print("=" * 60)

if __name__ == "__main__":
    setup_database()
