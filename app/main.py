import os
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    FAISS_PERSIST_DIR, SQLITE_PATH, UPLOAD_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MAX_AGENT_STEPS, TAVILY_API_KEY,
    TRACE_DIR,
)
from app.db.sqlite import Database
from app.db.vector import VectorStore
from app.services.llm import LLMService, _llm_service
from app.services.document import parse_file, chunk_text
from app.services.agent import run_agent
from app.services.tracer import Tracer
from app.tools.search_docs import set_vector_store
from app.models.schemas import (
    AskRequest, AskResponse, DocumentResponse, ConversationResponse, ErrorResponse,
)


_db: Database | None = None
_vector_store: VectorStore | None = None
_tracer: Tracer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _vector_store, _llm_service, _tracer

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    os.makedirs(TRACE_DIR, exist_ok=True)

    _db = Database(SQLITE_PATH)
    _vector_store = VectorStore(FAISS_PERSIST_DIR, collection_name="documents")
    _llm_service = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    _tracer = Tracer(TRACE_DIR)

    set_vector_store(_vector_store)

    yield

    if _vector_store:
        _vector_store.close()
    if _db:
        _db.close()


app = FastAPI(title="AI Research Agent", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    conversation_id = req.conversation_id or _db.create_conversation(req.question[:50])
    history = _db.get_conversation_messages(conversation_id) if req.conversation_id else []

    answer, steps, sources = run_agent(req.question, history=history, tracer=_tracer)

    for s in steps:
        _db.add_step(
            conv_id=conversation_id,
            step_number=s.step,
            thought=s.thought,
            tool_name=s.tool,
            tool_input=json.dumps(s.tool_input, ensure_ascii=False) if s.tool_input else None,
            tool_result=s.tool_result,
        )

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    _db.save_conversation_messages(conversation_id, history)

    return AskResponse(
        answer=answer,
        steps=steps,
        sources=sources,
        conversation_id=conversation_id,
    )


@app.post("/documents/upload", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    text = parse_file(file_path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    doc_id = _db.insert_document(
        filename=file.filename,
        file_type=os.path.splitext(file.filename)[1].lower(),
        file_size=len(content),
        chunk_count=len(chunks),
    )

    if chunks:
        embeddings = [_llm_service.embed(chunk) for chunk in chunks]
        metadatas = [
            {"document_id": doc_id, "document_name": file.filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        _vector_store.add(chunk_ids, chunks, embeddings, metadatas)

    record = _db.get_document(doc_id)
    return DocumentResponse(
        id=record["id"],
        filename=record["filename"],
        file_type=record["file_type"],
        file_size=record["file_size"],
        chunk_count=record["chunk_count"],
        created_at=record["created_at"],
    )


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents():
    records = _db.list_documents()
    return [
        DocumentResponse(
            id=r["id"], filename=r["filename"], file_type=r["file_type"],
            file_size=r["file_size"], chunk_count=r["chunk_count"], created_at=r["created_at"],
        )
        for r in records
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    doc = _db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    _vector_store.delete_by_document_id(doc_id)
    _db.delete_document(doc_id)
    return {"detail": "已删除"}


@app.get("/conversations", response_model=list[ConversationResponse])
def list_conversations():
    records = _db.list_conversations()
    return [
        ConversationResponse(
            id=r["id"], title=r["title"],
            step_count=r["step_count"], created_at=r["created_at"],
        )
        for r in records
    ]


@app.get("/conversations/{conv_id}", response_model=AskResponse)
def get_conversation(conv_id: str):
    conv = _db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    steps_raw = _db.get_steps_by_conversation(conv_id)
    steps = [
        AgentStep(
            step=s["step_number"],
            thought=s["thought"],
            tool=s["tool_name"],
            tool_input=json.loads(s["tool_input"]) if s["tool_input"] else None,
            tool_result=s["tool_result"],
        )
        for s in steps_raw
    ]
    return AskResponse(
        answer="", steps=steps, sources=[], conversation_id=conv_id,
    )


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    conv = _db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    _db.delete_conversation(conv_id)
    return {"detail": "已删除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
