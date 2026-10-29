import os
import shutil
from pathlib import Path
from typing import List, Tuple, Any, Optional

try:
    import spaces
    gpu_decorator = spaces.GPU
except (ImportError, Exception):
    def gpu_decorator(fn):
        return fn

import gradio as gr

from src.paths import resolve_path
from src.dataloader import load_file
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

upload_dir = resolve_path("data/uploads", "data/uploads")


@gpu_decorator
def query_rag(
    message: str,
    session_rag: RAGSearch,
    top_k: int,
) -> Tuple[str, str]:
    if not message.strip():
        return "", "Please enter a valid question."

    res = session_rag.search_with_context(message, top_k=int(top_k))
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


def _render_progress_html(percent: int, status_text: str, is_done: bool = False, is_error: bool = False) -> str:
    if is_error:
        accent_color = "#ef4444"
        bar_bg = "linear-gradient(90deg, #ef4444, #dc2626)"
        box_bg = "rgba(239, 68, 68, 0.1)"
        box_border = "rgba(239, 68, 68, 0.35)"
        text_color = "#ef4444"
    elif is_done:
        accent_color = "#10b981"
        bar_bg = "linear-gradient(90deg, #10b981, #059669)"
        box_bg = "rgba(16, 185, 129, 0.1)"
        box_border = "rgba(16, 185, 129, 0.35)"
        text_color = "#10b981"
    else:
        accent_color = "#3b82f6"
        bar_bg = "linear-gradient(90deg, #3b82f6, #6366f1)"
        box_bg = "rgba(59, 130, 246, 0.1)"
        box_border = "rgba(59, 130, 246, 0.35)"
        text_color = "var(--body-text-color, #1e293b)"

    return f"""
    <div style="background: {box_bg}; border: 1.5px solid {box_border}; border-radius: 10px; padding: 14px 18px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span class="progress-text" style="font-weight: 700; font-size: 0.95rem; color: {text_color}; display: flex; align-items: center; gap: 6px;">
                {status_text}
            </span>
            <span style="font-weight: 800; font-size: 0.95rem; color: {accent_color};">
                {percent}%
            </span>
        </div>
        <div style="width: 100%; background: rgba(156, 163, 175, 0.25); border-radius: 9999px; height: 10px; overflow: hidden;">
            <div style="width: {percent}%; background: {bar_bg}; height: 100%; border-radius: 9999px; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """


def start_upload_progress(files: List[Any]) -> str:
    if not files:
        return _render_progress_html(0, "⚠️ No files selected. Please select documents to upload.", is_done=False, is_error=True)
    return _render_progress_html(10, "⏳ Uploading documents and preparing storage...", is_done=False)


@gpu_decorator
def upload_and_index(files: List[Any], session_rag: RAGSearch):
    if not files:
        yield _render_progress_html(0, "⚠️ No files selected. Please select documents to upload.", is_done=False, is_error=True), session_rag
        return

    yield _render_progress_html(15, "⏳ Uploading documents and preparing storage...", is_done=False), session_rag

    upload_dir.mkdir(parents=True, exist_ok=True)
    allowed = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}
    total_files = len(files)
    new_docs = []
    saved_count = 0

    for idx, temp_file in enumerate(files, 1):
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

        loaded = load_file(str(dest))
        if loaded:
            new_docs.extend(loaded)

        pct = 15 + int(35 * (idx / total_files))
        yield _render_progress_html(pct, f"⏳ Processed {path_obj.name} ({saved_count}/{total_files})...", is_done=False), session_rag

    if not new_docs:
        yield _render_progress_html(0, "⚠️ No valid text could be extracted from uploaded document(s).", is_done=False, is_error=True), session_rag
        return

    yield _render_progress_html(75, f"⏳ Generating embeddings for {len(new_docs)} document sections...", is_done=False), session_rag
    session_rag.vectorstore.add_documents(new_docs)

    total_chunks = session_rag.vectorstore.index.ntotal if session_rag.vectorstore.index else 0
    yield _render_progress_html(
        100,
        f"✅ Successfully uploaded {saved_count} file(s) and indexed {total_chunks} document chunks in this session.",
        is_done=True,
    ), session_rag


def get_index_status(session_rag: Optional[RAGSearch] = None) -> str:
    if session_rag is None or session_rag.vectorstore.index is None or getattr(session_rag.vectorstore.index, "ntotal", 0) == 0:
        return "🟡 **Session Index empty** — upload documents below to begin asking questions."
    sources = {Path(m.get("source", "")).name for m in session_rag.vectorstore.metadata if m.get("source")}
    chunks = session_rag.vectorstore.index.ntotal
    return f"🟢 **Session Index active** ({len(sources)} file(s), {chunks} document chunks indexed)"


def create_ui() -> gr.Blocks:
    custom_css = """
    .gradio-container { max-width: 1000px !important; margin: auto !important; }
    .status-box { font-size: 0.9em; padding: 8px 12px; border-radius: 6px; }
    .dark .progress-text { color: #f8fafc !important; }
    """

    with gr.Blocks(title="Document RAG Assistant") as demo:
        session_rag = gr.State(RAGSearch)

        gr.Markdown(
            """
            # 📚 Document RAG Assistant
            Ask questions about your uploaded documents. Powered by **Sentence-Transformers**, **FAISS Vector Search**, and **Google Gemini LLM**.
            """
        )

        with gr.Row():
            status_markdown = gr.Markdown(value=get_index_status())
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            refresh_btn.click(fn=get_index_status, inputs=[session_rag], outputs=[status_markdown], show_progress="hidden")

        with gr.Tabs():
            with gr.Tab("💬 Ask & Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the key points in the uploaded document?",
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
                    inputs=[query_input, session_rag, top_k_slider],
                    outputs=[answer_output, citations_output],
                    show_progress="hidden",
                )
                query_input.submit(
                    fn=query_rag,
                    inputs=[query_input, session_rag, top_k_slider],
                    outputs=[answer_output, citations_output],
                    show_progress="hidden",
                )

            with gr.Tab("📂 Upload & Manage Documents"):
                gr.Markdown(
                    """
                    ### Upload Documents
                    Supported formats: **PDF, TXT, CSV, XLSX, DOCX, JSON**.
                    Files will be indexed exclusively for this session and available immediately for search.
                    """
                )
                file_upload = gr.File(
                    label="Select or Drag & Drop Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"],
                )
                upload_btn = gr.Button("📤 Upload & Rebuild Index", variant="primary")
                upload_status = gr.HTML(value="")

                upload_btn.click(
                    fn=start_upload_progress,
                    inputs=[file_upload],
                    outputs=[upload_status],
                    show_progress="hidden",
                ).then(
                    fn=upload_and_index,
                    inputs=[file_upload, session_rag],
                    outputs=[upload_status, session_rag],
                    show_progress="hidden",
                ).then(
                    fn=get_index_status,
                    inputs=[session_rag],
                    outputs=[status_markdown],
                    show_progress="hidden",
                )

        gr.Markdown(
            """
            ---
            *API documentation & REST endpoints are available at [`/docs`](/docs). Built for Hugging Face Spaces.*
            """
        )

        demo.queue()

    return demo
