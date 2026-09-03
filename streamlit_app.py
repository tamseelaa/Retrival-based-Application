import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


# --- Sidebar: Direct LLM Healthcheck (No DB required) ---
with st.sidebar:
    st.header("⚙️ Diagnostics")
    st.caption("Test Ollama LLM directly without database")
    test_prompt = st.text_input("Test prompt", value="Hello! Reply with a short confirmation that you are working.", key="sidebar_prompt")
    if st.button("🧪 Test Local LLM"):
        with st.spinner("Connecting to local Ollama LLM..."):
            try:
                from openai import OpenAI
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                api_key = os.getenv("OLLAMA_API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
                model = os.getenv("LLM_MODEL", "llama3.1:8b")
                client = OpenAI(base_url=base_url, api_key=api_key)

                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=150,
                    temperature=0.2,
                )
                elapsed = time.time() - t0
                reply = resp.choices[0].message.content.strip()
                st.success(f"✅ Connected to `{model}` in {elapsed:.2f}s!")
                st.write(f"**Response:** {reply}")
            except Exception as e:
                st.error(f"❌ Failed to reach Ollama: {e}")
                st.info("Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.1:8b`).")



def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                err = run.get("error")
                if not err:
                    out = run.get("output")
                    if isinstance(out, dict):
                        err = out.get("error")
                if isinstance(err, dict):
                    err_msg = err.get("message") or err.get("name") or str(err)
                elif err:
                    err_msg = str(err)
                else:
                    err_msg = f"Function run {status}"
                raise RuntimeError(err_msg)
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


def extract_answer_and_sources(raw_output) -> tuple[str, list]:
    if not raw_output:
        return "", []
    if isinstance(raw_output, str):
        try:
            raw_output = json.loads(raw_output)
        except Exception:
            return raw_output, []
    if isinstance(raw_output, dict):
        if "answer" in raw_output:
            return str(raw_output.get("answer") or ""), raw_output.get("sources") or []
        if "data" in raw_output and isinstance(raw_output["data"], dict):
            return str(raw_output["data"].get("answer") or ""), raw_output["data"].get("sources") or []
        if "output" in raw_output and isinstance(raw_output["output"], dict):
            return str(raw_output["output"].get("answer") or ""), raw_output["output"].get("sources") or []
        if "result" in raw_output:
            return str(raw_output.get("result") or ""), []
    return str(raw_output), []


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            try:
                # Fire-and-forget event to Inngest for observability/workflow
                event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
                # Poll the local Inngest API for the run's output
                output = wait_for_run_output(event_id)
                answer, sources = extract_answer_and_sources(output)

                st.subheader("Answer")
                if answer:
                    st.write(answer)
                else:
                    st.write("(No answer returned)")
                    if output:
                        st.caption("Raw run output:")
                        st.json(output)

                if sources:
                    st.caption("Sources")
                    for s in sources:
                        st.write(f"- {s}")
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                st.info("💡 Tip: Check your FastAPI terminal for detailed step error logs.")

