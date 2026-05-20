from pathlib import Path
from typing import List, Optional
import shutil
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.dataloader import load_all_documents
from src.paths import resolve_path
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}

app = FastAPI(title="RAG API")

data_dir = resolve_path(None, "data")
upload_dir = resolve_path("data/uploads", "data/uploads")
persist_dir = resolve_path(None, "data/faiss_store")

upload_dir.mkdir(parents=True, exist_ok=True)

rag_lock = threading.Lock()
rag_search: Optional[RAGSearch] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    summary: str
    results: List[dict]


def _unique_path(directory: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    base = Path(safe_name).stem
    suffix = Path(safe_name).suffix
    candidate = directory / safe_name
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}_{counter}{suffix}"
        counter += 1
    return candidate


def _get_rag() -> RAGSearch:
    global rag_search
    if rag_search is None:
        try:
            rag_search = RAGSearch(persist_dir=str(persist_dir), data_dir=str(data_dir))
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return rag_search


def _rebuild_index() -> None:
    docs = load_all_documents(str(data_dir))
    if not docs:
        raise HTTPException(status_code=400, detail="No documents found to build the index.")
    vectorstore = FaissVectorStore(persist_dir=str(persist_dir))
    vectorstore.build_from_documents(docs)
    global rag_search
    rag_search = RAGSearch(persist_dir=str(persist_dir), data_dir=str(data_dir))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), rebuild_index: bool = True) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    saved_files: List[str] = []
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        destination = _unique_path(upload_dir, file.filename)
        with destination.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
        await file.close()
        saved_files.append(destination.name)

    if rebuild_index:
        with rag_lock:
            _rebuild_index()

    return {"saved": saved_files, "reindexed": rebuild_index}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    with rag_lock:
        rag = _get_rag()
        result = rag.search_with_context(request.query, top_k=request.top_k)
    return QueryResponse(query=request.query, summary=result["summary"], results=result["results"])
