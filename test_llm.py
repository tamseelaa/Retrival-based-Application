import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def test_ollama():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
    model = os.getenv("LLM_MODEL", "llama3.1:8b")

    print(f"Connecting to Ollama at: {base_url}")
    print(f"Testing LLM Model: {model}")
    print("-" * 50)

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: 'openai' package not installed. Run 'pip install openai' or 'uv sync'.")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    prompt = "Hello! Please reply in one short sentence confirming that you are working."
    print(f"Prompt: {prompt}\n")

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100,
        )
        elapsed = time.time() - start_time
        reply = response.choices[0].message.content.strip()

        print(f"Response ({elapsed:.2f}s):")
        print(f"  \"{reply}\"")
        print("✅ SUCCESS: Your local Ollama LLM is working properly!")
    except Exception as e:
        print(f"❌ FAILED to connect to Ollama / model '{model}'.")
        print(f"Error details: {e}\n")
        print("Troubleshooting:")
        print("1. Is Ollama running? Run: ollama serve")
        print(f"2. Is the model downloaded? Run: ollama pull {model}")

    # Test Embedding Model
    print("\n" + "=" * 50)
    print("Testing Embedding Model...")
    from data_loader import EMBED_MODEL, EMBED_DIM, embed_texts
    print(f"Model: {EMBED_MODEL} (Expected Dim: {EMBED_DIM})")
    try:
        t0 = time.time()
        vecs = embed_texts(["Test embedding sentence for Ollama."])
        t_embed = time.time() - t0
        actual_dim = len(vecs[0]) if vecs else 0
        print(f"Embedding Generated ({t_embed:.2f}s, Dimension: {actual_dim})")
        if actual_dim == EMBED_DIM:
            print("✅ SUCCESS: Embedding model is working properly!")
        else:
            print(f"⚠️ Warning: Embedding dimension ({actual_dim}) != configured ({EMBED_DIM}). Run 'python setup_db.py'.")
    except Exception as e:
        print(f"❌ FAILED to generate embeddings: {e}")
        print(f"💡 Solution: Run 'ollama pull {EMBED_MODEL}' in your terminal.")

    print("=" * 50)

if __name__ == "__main__":
    test_ollama()
