import os
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.services.metrics import record_usage

from app.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    FAISS_PERSIST_DIR, SQLITE_PATH, UPLOAD_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    TRACE_DIR, MCP_SERVERS,
    VISION_API_KEY, VISION_BASE_URL, VISION_MODEL,
)
from app.db.sqlite import Database
from app.db.vector import VectorStore
from app.services.llm import AsyncLLMService, LLMService
from app.services.document import parse_file, chunk_text
from app.services.agent import run_agent, set_mcp_tools
from app.services.memory import SemanticMemory, set_memory
from app.services.tracer import Tracer
from app.services.vision import VisionService, VisionError, VisionUnavailableError
from app.tools.search_docs import set_vector_store
from app.models.schemas import (
    AgentStep, AskRequest, AskResponse, DocumentResponse, ConversationResponse,
)


_db: Database | None = None
_vector_store: VectorStore | None = None
_tracer: Tracer | None = None
_mcp_manager = None
_llm_service = None
_vision_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _vector_store, _tracer, _mcp_manager, _llm_service, _vision_service

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    os.makedirs(TRACE_DIR, exist_ok=True)

    _db = Database(SQLITE_PATH)
    _vector_store = VectorStore(FAISS_PERSIST_DIR, collection_name="documents")
    _tracer = Tracer(TRACE_DIR)

    set_vector_store(_vector_store)

    # ── v1 遗留 P0 修复 ──────────────────────────────────────────────────
    # agent.py 第 7 行 `from app.services.llm import _llm_service` 是 import 时的
    # 值绑定（当时为 None），而下面 main 自己的 global 赋值只写本模块命名空间，
    # 两者互不相通——v1 真实运行 /ask 必然 None.chat_with_tools 崩溃，
    # 30 个全 mock 测试从未暴露。必须在 lifespan 里逐模块回填单例。
    import app.services.llm as llm_module
    import app.services.agent as agent_module
    _llm_service = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    llm_module._llm_service = _llm_service
    agent_module._llm_service = _llm_service
    llm_module._async_llm_service = AsyncLLMService(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
    )

    set_memory(SemanticMemory(_db, embed_fn=_llm_service.embed))
    # 显式传参：VisionService 默认值在 vision.py import 时冻结，patch config 属性无效
    _vision_service = VisionService(api_key=VISION_API_KEY, base_url=VISION_BASE_URL,
                                    model=VISION_MODEL)

    # MCP stdio 子进程：启动拉起 / shutdown 终止（孤儿检测在 provider 内）
    _mcp_manager = None
    if MCP_SERVERS.strip():
        from app.mcp_client.provider import MCPProviderManager, parse_servers_spec
        _mcp_manager = MCPProviderManager(parse_servers_spec(MCP_SERVERS))
        await _mcp_manager.start_all()
        set_mcp_tools(_mcp_manager.get_tools())

    yield

    if _mcp_manager:
        await _mcp_manager.shutdown_all()
        set_mcp_tools({})
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


from app.api.research import router as research_router  # noqa: E402  # 多 agent 端点
app.include_router(research_router)


@app.get("/health")
def health():
    """存活探针 + 通道健康态：无鉴权零依赖，供 Docker HEALTHCHECK 和运维查看。"""
    vision_status = _vision_service.status() if _vision_service else {"configured": False}
    return {
        "status": "ok",
        "mcp_enabled": _mcp_manager is not None,
        "vision": vision_status,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    conversation_id = req.conversation_id or _db.create_conversation(req.question[:50])
    history = _db.get_conversation_messages(conversation_id) if req.conversation_id else []

    # ── 意图识别路由层：chat 直答 / tool_research 单 agent / deep_research 多 agent ──
    if req.force_route in ("chat", "tool_research", "deep_research"):
        route, route_via = req.force_route, "forced"
    else:
        from app.services.intent_router import classify_intent
        decision = classify_intent(req.question, history_len=len(history))
        route, route_via = decision["intent"], decision["via"]

    metrics: dict = {"route_via": route_via}
    answer, steps, sources = "", [], []

    if route == "chat":
        start = time.time()
        answer = _llm_service.chat([
            {"role": "system", "content": "你是友好的对话助手，直接简洁地回答，不需要调用工具。"},
            *[{"role": h["role"], "content": h["content"]} for h in history[-6:]],
            {"role": "user", "content": req.question},
        ])
        record_usage(metrics, getattr(_llm_service, "last_usage", None))
        metrics["latency_s"] = round(time.time() - start, 2)

    elif route == "deep_research":
        import asyncio
        from app.api.research import _run_multi_agent
        from app.multi_agent.schemas import TaskRequest

        start = time.time()
        result = asyncio.run(_run_multi_agent(
            TaskRequest(task=req.question, max_iterations=2)
        ))
        metrics["latency_s"] = round(time.time() - start, 2)
        metrics["iterations"] = result.iterations
        metrics["passed"] = result.passed
        answer = result.report

    else:  # tool_research（默认兜底路径）
        start = time.time()
        sink: dict = {}
        answer, steps, sources = run_agent(
            req.question, history=history, tracer=_tracer, metrics_sink=sink
        )
        metrics.update(sink)
        metrics["latency_s"] = round(time.time() - start, 2)

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
        route=route,
        metrics=metrics,
    )


@app.get("/stats/{conversation_id}")
def conversation_stats(conversation_id: str):
    """会话级工具调用聚合：次数/成败率，数据源 agent_steps 表。"""
    conv = _db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")

    steps = _db.get_steps_by_conversation(conversation_id)
    by_tool: dict[str, dict] = {}
    total = 0
    for s in steps:
        if not s["tool_name"]:
            continue  # answer 步不是工具调用
        ok = not (s["tool_result"] or "").startswith('{"error"')
        stats = by_tool.setdefault(s["tool_name"], {"count": 0, "success": 0, "fail": 0})
        stats["count"] += 1
        stats["success" if ok else "fail"] += 1
        total += 1

    fail = sum(v["fail"] for v in by_tool.values())
    return {
        "conversation_id": conversation_id,
        "steps_total": len(steps),
        "tool_calls_total": total,
        "tool_success_rate": round((total - fail) / total, 3) if total else None,
        "by_tool": by_tool,
    }


@app.post("/documents/upload", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)):
    # filename 来自 multipart、用户可控：剥路径分量防穿越（与 rag-qa v2 同款加固）
    raw_name = (file.filename or "").replace("\\", "/").strip()
    safe_name = os.path.basename(raw_name)
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法文件名")
    # 图片走视觉专线单独处理，不进 parse_file 的文本白名单
    ext = os.path.splitext(safe_name)[1].lower()

    file_path = os.path.join(UPLOAD_DIR, safe_name)

    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # ── 多模态专线：图片走视觉描述后入库（文本模型没有视觉能力） ──
    if ext in {".png", ".jpg", ".jpeg"}:
        try:
            description = _vision_service.describe_image(file_path)
        except VisionUnavailableError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except VisionError as e:
            raise HTTPException(status_code=502, detail=f"视觉专线调用失败: {e}")
        doc_id = _db.insert_document(
            filename=safe_name,
            file_type=ext.lstrip("."),
            file_size=len(content),
            chunk_count=1,
        )
        embedding = _llm_service.embed(description)
        _vector_store.add(
            [f"{doc_id}_img0"], [description], [embedding],
            [{"document_id": doc_id, "document_name": safe_name,
              "chunk_index": 0, "modality": "image"}],
        )
        record = _db.get_document(doc_id)
        return DocumentResponse(
            id=record["id"], filename=record["filename"], file_type=record["file_type"],
            file_size=record["file_size"], chunk_count=record["chunk_count"],
            created_at=record["created_at"],
        )

    text = parse_file(file_path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    doc_id = _db.insert_document(
        filename=safe_name,
        file_type=os.path.splitext(safe_name)[1].lower(),
        file_size=len(content),
        chunk_count=len(chunks),
    )

    if chunks:
        embeddings = [_llm_service.embed(chunk) for chunk in chunks]
        metadatas = [
            {"document_id": doc_id, "document_name": safe_name, "chunk_index": i}
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
