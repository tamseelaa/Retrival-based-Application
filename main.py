import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from openai import OpenAI
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import PgVectorStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


def _get_llm_client() -> OpenAI:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
    return OpenAI(base_url=base_url, api_key=api_key)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> dict:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return {"chunks": chunks, "source_id": source_id}

    def _upsert(data: dict) -> dict:
        chunks = data["chunks"] if isinstance(data, dict) else data.chunks
        source_id = data["source_id"] if isinstance(data, dict) else data.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        PgVectorStorage().upsert(ids, vecs, payloads)
        return {"ingested": len(chunks)}

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx))
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src))
    return ingested if isinstance(ingested, dict) else ingested.model_dump()


logger = logging.getLogger("uvicorn")


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> dict:
        try:
            logger.info(f"🔎 Embedding question: '{question}'")
            query_vec = embed_texts([question])[0]
            logger.info(f"🔎 Querying pgvector database (top_k={top_k})...")
            store = PgVectorStorage()
            found = store.search(query_vec, top_k)
            logger.info(f"✅ Found {len(found.get('contexts', []))} matching chunks")
            return {"contexts": found.get("contexts", []), "sources": found.get("sources", [])}
        except Exception as e:
            logger.error(f"❌ Error during vector search: {e}", exc_info=True)
            raise

    def _generate_answer(user_content: str) -> str:
        try:
            logger.info("🤖 Calling local Ollama LLM for answer generation...")
            client = _get_llm_client()
            model = os.getenv("LLM_MODEL", "llama3.1:8b")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You answer questions using only the provided context."},
                    {"role": "user", "content": user_content},
                ],
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            )
            ans = response.choices[0].message.content.strip()
            logger.info("✅ Ollama LLM generated answer successfully!")
            return ans
        except Exception as e:
            logger.error(f"❌ Error during Ollama generation: {e}", exc_info=True)
            raise

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))

    contexts = found["contexts"] if isinstance(found, dict) else getattr(found, "contexts", [])
    sources = found["sources"] if isinstance(found, dict) else getattr(found, "sources", [])

    context_block = "\n\n".join(f"- {c}" for c in contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    answer = await ctx.step.run("llm-answer", lambda: _generate_answer(user_content))

    return {"answer": answer, "sources": sources, "num_contexts": len(contexts)}

app = FastAPI(title="RAG Inngest Backend")

@app.get("/")
def health_check():
    return {"status": "ok", "app": "rag_app", "inngest_endpoint": "/api/inngest"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])