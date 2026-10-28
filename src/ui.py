import os
import shutil
from pathlib import Path
from typing import List, Tuple, Any
import gradio as gr

from src.paths import resolve_path
from src.dataloader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

data_dir = resolve_path(None, "data")
upload_dir = resolve_path("data/uploads", "data/uploads")
persist_dir = resolve_path(None, "data/faiss_store")

_rag_instance = None


def get_rag() -> RAGSearch:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSearch(persist_dir=str(persist_dir), data_dir=str(data_dir))
    return _rag_instance


def query_rag(message: str, history: List[Tuple[str, str]], top_k: int) -> Tuple[str, str]:
    if not message.strip():
        return "", "Please enter a valid question."

    rag = get_rag()
    res = rag.search_with_context(message, top_k=int(top_k))
    summary = res.get("summary", "No response generated.")
    results = res.get("results", [])

    citations_md = "### 📑 Retrieved Document Contexts\n\n"
    if not results:
        citations_md += "_No relevant contexts retrieved._"
    else:
        for idx, item in enumerate(results, 1):
            source = Path(item.get("source", "Unknown")).name
            page = item.get("page")
            page_str = f" (Page {page + 1})" if page is not None and isinstance(page, int) else ""
            score = item.get("distance")
            score_str = f"Score: {score:.4f}" if isinstance(score, (int, float)) else ""
            text = (item.get("text") or "").strip().replace("\n", " ")
            if len(text) > 300:
                text = text[:300] + "..."
            citations_md += f"**{idx}. `{source}`{page_str}** — _{score_str}_\n> {text}\n\n"

    return summary, citations_md


def upload_and_index(files: List[Any]) -> str:
    if not files:
        return "⚠️ No files selected."

    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    allowed = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}

    for temp_file in files:
        file_path = getattr(temp_file, "name", None) or str(temp_file)
        path_obj = Path(file_path)
        if path_obj.suffix.lower() not in allowed:
            continue
        dest = upload_dir / path_obj.name
        counter = 1
        while dest.exists():
            dest = upload_dir / f"{path_obj.stem}_{counter}{path_obj.suffix}"
            counter += 1
        shutil.copy2(file_path, dest)
        saved_count += 1

    if saved_count == 0:
        return "⚠️ No valid documents with supported formats (.pdf, .txt, .csv, .xlsx, .docx, .json) were found."

    # Rebuild index
    docs = load_all_documents(str(data_dir))
    if not docs:
        return f"Saved {saved_count} file(s), but no valid text could be extracted."

    store = FaissVectorStore(persist_dir=str(persist_dir))
    store.build_from_documents(docs)

    global _rag_instance
    _rag_instance = RAGSearch(persist_dir=str(persist_dir), data_dir=str(data_dir))

    return f"✅ Successfully uploaded {saved_count} file(s) and reindexed {len(docs)} document chunks!"


def get_index_status() -> str:
    faiss_file = persist_dir / "faiss.index"
    meta_file = persist_dir / "metadata.pkl"
    if faiss_file.exists() and meta_file.exists():
        size_kb = faiss_file.stat().st_size / 1024
        return f"🟢 **FAISS Index active** (`faiss.index`: {size_kb:.1f} KB)"
    return "🟡 **Index not found** — upload files or run ingestion to create."


def create_ui() -> gr.Blocks:
    custom_css = """
    .gradio-container { max-width: 1000px !important; margin: auto !important; }
    .status-box { font-size: 0.9em; padding: 8px 12px; border-radius: 6px; }
    """

    with gr.Blocks(title="Document RAG Assistant") as demo:
        gr.Markdown(
            """
            # 📚 Document RAG Assistant
            Ask questions about your uploaded documents. Powered by **Sentence-Transformers**, **FAISS Vector Search**, and **OpenRouter LLM**.
            """
        )

        with gr.Row():
            status_markdown = gr.Markdown(value=get_index_status())
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            refresh_btn.click(fn=get_index_status, inputs=[], outputs=[status_markdown])

        with gr.Tabs():
            with gr.Tab("💬 Ask & Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the frontend and backend technologies used in EventX?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1, label="Top Context Chunks (k)"
                        )
                        ask_btn = gr.Button("🔍 Search & Answer", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Markdown(
                            label="Answer",
                            value="*Your AI-generated answer will appear here.*",
                        )
                    with gr.Column(scale=1):
                        citations_output = gr.Markdown(
                            label="Citations & Sources",
                            value="*Retrieved document excerpts will appear here.*",
                        )

                ask_btn.click(
                    fn=query_rag,
                    inputs=[query_input, gr.State([]), top_k_slider],
                    outputs=[answer_output, citations_output],
                )
                query_input.submit(
                    fn=query_rag,
                    inputs=[query_input, gr.State([]), top_k_slider],
                    outputs=[answer_output, citations_output],
                )

            with gr.Tab("📂 Upload & Manage Documents"):
                gr.Markdown(
                    """
                    ### Upload Documents
                    Supported formats: **PDF, TXT, CSV, XLSX, DOCX, JSON**.
                    Files will be added to the document repository and automatically embedded into the FAISS index.
                    """
                )
                file_upload = gr.File(
                    label="Select or Drag & Drop Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"],
                )
                upload_btn = gr.Button("📤 Upload & Rebuild Index", variant="primary")
                upload_status = gr.Markdown(value="")

                upload_btn.click(
                    fn=upload_and_index,
                    inputs=[file_upload],
                    outputs=[upload_status],
                ).then(
                    fn=get_index_status,
                    inputs=[],
                    outputs=[status_markdown],
                )

        gr.Markdown(
            """
            ---
            *API documentation & REST endpoints are available at [`/docs`](/docs). Built for Hugging Face Spaces.*
            """
        )

    return demo
