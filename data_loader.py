import os
import requests
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

def _has_valid_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and not key.startswith("your_openai") and not key.startswith("sk-your"))

_configured_model = os.getenv("EMBED_MODEL")
if not _has_valid_openai_key() and (_configured_model is None or _configured_model.startswith("text-embedding")):
    EMBED_MODEL = "nomic-embed-text"
    EMBED_DIM = 768
else:
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large" if _has_valid_openai_key() else "nomic-embed-text")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "3072" if _has_valid_openai_key() else "768"))

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def _embed_ollama_native(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    """Generate embeddings using Ollama's native /api/embed and /api/embeddings endpoints."""
    base_host = base_url.rstrip("/").removesuffix("/v1")

    # 1. Try Ollama native /api/embed (batch endpoint)
    try:
        resp = requests.post(
            f"{base_host}/api/embed",
            json={"model": model, "input": texts},
            timeout=120
        )
        if resp.status_code == 200:
            data = resp.json()
            if "embeddings" in data and data["embeddings"]:
                return data["embeddings"]
    except Exception:
        pass

    # 2. Try Ollama /api/embeddings (single-item endpoint)
    try:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{base_host}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60
            )
            if resp.status_code == 200:
                embeddings.append(resp.json()["embedding"])
            else:
                resp.raise_for_status()
        if len(embeddings) == len(texts):
            return embeddings
    except Exception:
        pass

    # 3. Fallback to /v1/embeddings via OpenAI client
    client = OpenAI(base_url=f"{base_host}/v1", api_key="ollama")
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for text chunks using either OpenAI or Ollama."""
    if not texts:
        return []

    # If OpenAI key or explicit external embedding URL is configured
    if _has_valid_openai_key() and not os.getenv("EMBED_BASE_URL"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # Otherwise use local Ollama
    ollama_url = os.getenv("EMBED_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return _embed_ollama_native(texts, ollama_url, EMBED_MODEL)