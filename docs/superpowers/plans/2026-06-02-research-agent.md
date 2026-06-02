# Research Agent 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零实现一个具备 3 个工具（文档检索、联网搜索、计算器）的 AI Agent 系统，核心循环不用任何框架。

**Architecture:** FastAPI 三层架构 → Agent 引擎 (思考-执行-观察循环) → 工具集 + 记忆模块 → FAISS/SQLite 数据层。前端为现代科技感聊天界面。

**Tech Stack:** FastAPI, DeepSeek API (Function Calling), FAISS + fastembed, SQLite, Tavily Search API, ui-ux-pro-max 生成前端

**代码分工:**
- 用户编写: 工具定义 (Task 5/6/7)、Agent 核心循环 + Prompt (Task 10)
- Claude 编写: 项目脚手架、数据层、API 层、文档解析、Web 前端、Docker

---

### Task 1: 项目脚手架

**Files:**
- Create: `C:\Users\LX\projects\02-research-agent\.env`
- Create: `C:\Users\LX\projects\02-research-agent\.gitignore`
- Create: `C:\Users\LX\projects\02-research-agent\requirements.txt`
- Create: `C:\Users\LX\projects\02-research-agent\app\__init__.py`

- [ ] **Step 1: 创建 .env 配置文件**

```bash
cat > C:\Users\LX\projects\02-research-agent\.env << 'EOF'
DEEPSEEK_API_KEY=sk-c2a3d3dd3fb74eb4806dac1af2ce6f0b
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
FAISS_PERSIST_DIR=./data/faiss
SQLITE_PATH=./data/research.db
UPLOAD_DIR=./data/uploads
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
MAX_AGENT_STEPS=10
TAVILY_API_KEY=your_tavily_key_here
EOF
```

- [ ] **Step 2: 创建 .gitignore**

```bash
cat > C:\Users\LX\projects\02-research-agent\.gitignore << 'EOF'
__pycache__/
*.pyc
.env
data/faiss/
data/uploads/*
!data/uploads/.gitkeep
*.db
*.db-shm
*.db-wal
.venv/
.pytest_cache/
EOF
```

- [ ] **Step 3: 创建 requirements.txt**

```bash
cat > C:\Users\LX\projects\02-research-agent\requirements.txt << 'EOF'
fastapi==0.115.6
uvicorn[standard]==0.34.0
openai==1.58.1
faiss-cpu==1.9.0
fastembed==0.4.1
PyPDF2==3.0.1
python-docx==1.1.2
python-dotenv==1.0.1
python-multipart==0.0.19
pytest==8.3.4
httpx==0.28.1
numpy==1.26.4
requests==2.32.3
EOF
```

- [ ] **Step 4: 创建 app/__init__.py**

```
# empty file
```

- [ ] **Step 5: 创建 data 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\data\uploads
touch C:\Users\LX\projects\02-research-agent\data\uploads\.gitkeep
```

- [ ] **Step 6: 安装依赖并验证**

```bash
cd C:\Users\LX\projects\02-research-agent && pip install -r requirements.txt -q
```

Expected: 所有包安装成功，无报错。

- [ ] **Step 7: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git init && git add -A && git commit -m "chore: project scaffolding"
```

---

### Task 2: 配置模块

**Files:**
- Create: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_config.py
import os


def test_config_reads_env_vars():
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["TAVILY_API_KEY"] = "test-tavily"
    os.environ["MAX_AGENT_STEPS"] = "5"

    from app.config import DEEPSEEK_API_KEY, TAVILY_API_KEY, MAX_AGENT_STEPS

    assert DEEPSEEK_API_KEY == "test-key"
    assert TAVILY_API_KEY == "test-tavily"
    assert MAX_AGENT_STEPS == 5


def test_config_defaults():
    # 清除环境变量，测试默认值
    for key in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K"]:
        os.environ.pop(key, None)

    import importlib
    import app.config
    importlib.reload(app.config)

    assert app.config.CHUNK_SIZE == 500
    assert app.config.CHUNK_OVERLAP == 50
    assert app.config.TOP_K == 3
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: 写 config.py**

```python
# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./data/faiss")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/research.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_config.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/config.py tests/test_config.py && git commit -m "feat: add config module"
```

---

### Task 3: SQLite 数据库模块

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/sqlite.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: 创建 db 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\db
touch C:\Users\LX\projects\02-research-agent\app\db\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_db.py
import os
import pytest
from app.db.sqlite import Database


@pytest.fixture
def db():
    path = "./data/test_research.db"
    database = Database(path)
    yield database
    database.close()
    if os.path.exists(path):
        os.remove(path)
    for ext in ["-shm", "-wal"]:
        if os.path.exists(path + ext):
            os.remove(path + ext)


def test_insert_and_get_document(db):
    doc_id = db.insert_document("test.pdf", "pdf", 1024, 5)
    assert doc_id

    doc = db.get_document(doc_id)
    assert doc["filename"] == "test.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["chunk_count"] == 5


def test_list_documents(db):
    db.insert_document("a.pdf", "pdf", 100, 2)
    db.insert_document("b.txt", "txt", 200, 1)
    docs = db.list_documents()
    assert len(docs) == 2


def test_delete_document(db):
    doc_id = db.insert_document("x.pdf", "pdf", 50, 1)
    db.delete_document(doc_id)
    assert db.get_document(doc_id) is None


def test_create_and_get_conversation(db):
    conv_id = db.create_conversation("测试对话")
    assert conv_id

    conv = db.get_conversation(conv_id)
    assert conv["title"] == "测试对话"


def test_list_conversations(db):
    db.create_conversation("对话1")
    db.create_conversation("对话2")
    convs = db.list_conversations()
    assert len(convs) == 2


def test_delete_conversation(db):
    conv_id = db.create_conversation("待删除")
    db.delete_conversation(conv_id)
    assert db.get_conversation(conv_id) is None


def test_add_and_get_agent_steps(db):
    conv_id = db.create_conversation("测试")
    db.add_step(conv_id, 1, "搜索一下", "search_docs", '{"query":"test"}', '["结果1"]')
    db.add_step(conv_id, 2, "信息够了", None, None, None)

    steps = db.get_steps_by_conversation(conv_id)
    assert len(steps) == 2
    assert steps[0]["thought"] == "搜索一下"
    assert steps[1]["tool_name"] is None


def test_memories_crud(db):
    db.save_memory("用户偏好", "喜欢表格", "conv-1")
    memories = db.get_memories()
    assert len(memories) == 1
    assert memories[0]["key"] == "用户偏好"

    mid = memories[0]["id"]
    db.delete_memory(mid)
    assert len(db.get_memories()) == 0
```

- [ ] **Step 3: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: 写 sqlite.py**

```python
# app/db/sqlite.py
import sqlite3
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _uid():
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                thought TEXT,
                tool_name TEXT,
                tool_input TEXT,
                tool_result TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_conversation_id TEXT,
                created_at TEXT NOT NULL
            );
        """)

    def close(self):
        self.conn.close()

    # ── documents ──

    def insert_document(self, filename: str, file_type: str, file_size: int, chunk_count: int) -> str:
        doc_id = _uid()
        self.conn.execute(
            "INSERT INTO documents (id, filename, file_type, file_size, chunk_count, created_at) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, file_type, file_size, chunk_count, _now())
        )
        self.conn.commit()
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def delete_document(self, doc_id: str):
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()

    # ── conversations ──

    def create_conversation(self, title: str = "") -> str:
        conv_id = _uid()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)",
            (conv_id, title, _now())
        )
        self.conn.commit()
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,))
        row = cur.fetchone()
        if not row:
            return None
        conv = dict(row)
        cur = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM agent_steps WHERE conversation_id=?", (conv_id,)
        )
        conv["step_count"] = cur.fetchone()["cnt"]
        return conv

    def list_conversations(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        convs = [dict(r) for r in cur.fetchall()]
        for c in convs:
            cur = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM agent_steps WHERE conversation_id=?", (c["id"],)
            )
            c["step_count"] = cur.fetchone()["cnt"]
        return convs

    def delete_conversation(self, conv_id: str):
        self.conn.execute("DELETE FROM agent_steps WHERE conversation_id=?", (conv_id,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        self.conn.commit()

    # ── agent steps ──

    def add_step(self, conv_id: str, step_number: int, thought: str | None,
                 tool_name: str | None, tool_input: str | None, tool_result: str | None):
        self.conn.execute(
            "INSERT INTO agent_steps (conversation_id, step_number, thought, tool_name, tool_input, tool_result, created_at) VALUES (?,?,?,?,?,?,?)",
            (conv_id, step_number, thought, tool_name, tool_input, tool_result, _now())
        )
        self.conn.commit()

    def get_steps_by_conversation(self, conv_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM agent_steps WHERE conversation_id=? ORDER BY step_number ASC",
            (conv_id,)
        )
        return [dict(r) for r in cur.fetchall()]

    # ── memories ──

    def save_memory(self, key: str, value: str, source_conv_id: str | None = None):
        self.conn.execute(
            "INSERT INTO memories (key, value, source_conversation_id, created_at) VALUES (?,?,?,?)",
            (key, value, source_conv_id, _now())
        )
        self.conn.commit()

    def get_memories(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM memories ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def delete_memory(self, memory_id: int):
        self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
```

- [ ] **Step 5: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_db.py -v
```

Expected: 8 PASS

- [ ] **Step 6: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/db/sqlite.py tests/test_db.py && git commit -m "feat: add SQLite database module"
```

---

### Task 4: FAISS 向量存储

**Files:**
- Create: `app/db/vector.py`
- Test: `tests/test_vector.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_vector.py
import os
import shutil
import pytest
from app.db.vector import VectorStore


@pytest.fixture
def vs():
    path = "./data/test_faiss"
    store = VectorStore(path)
    yield store
    store.close()
    if os.path.exists(path):
        shutil.rmtree(path)


def test_add_and_search(vs):
    ids = ["chunk-1", "chunk-2"]
    texts = ["Python 是一门编程语言", "FAISS 是向量数据库"]
    embeddings = [[0.1]*384, [0.2]*384]
    metadatas = [
        {"document_id": "doc-1", "document_name": "test.pdf", "chunk_index": 0},
        {"document_id": "doc-1", "document_name": "test.pdf", "chunk_index": 1},
    ]

    vs.add(ids, texts, embeddings, metadatas)

    result = vs.search([0.1]*384, n_results=2)
    assert len(result["ids"][0]) == 2
    assert result["documents"][0][0] == "Python 是一门编程语言"


def test_search_empty_store(vs):
    result = vs.search([0.5]*384, n_results=3)
    assert result["ids"] == [[]]


def test_delete_by_document_id(vs):
    vs.add(["c1"], ["text1"], [[0.1]*384], [{"document_id": "doc-a", "document_name": "a.pdf", "chunk_index": 0}])
    vs.add(["c2"], ["text2"], [[0.2]*384], [{"document_id": "doc-b", "document_name": "b.pdf", "chunk_index": 0}])

    vs.delete_by_document_id("doc-a")
    result = vs.search([0.1]*384, n_results=3)
    assert len(result["ids"][0]) == 1


def test_persistence(vs):
    vs.add(["c1"], ["persist test"], [[0.3]*384], [{"document_id": "doc-1", "document_name": "p.pdf", "chunk_index": 0}])
    vs.close()

    store2 = VectorStore("./data/test_faiss")
    result = store2.search([0.3]*384, n_results=1)
    assert result["documents"][0][0] == "persist test"
    store2.close()
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_vector.py -v
```

Expected: FAIL

- [ ] **Step 3: 写 vector.py**

```python
# app/db/vector.py
import os
import json
import numpy as np
import faiss


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        os.makedirs(persist_dir, exist_ok=True)
        self.index_path = os.path.join(persist_dir, f"{collection_name}.index")
        self.data_path = os.path.join(persist_dir, f"{collection_name}.data.json")

        self.index: faiss.IndexFlatL2 | None = None
        self.records: list[dict] = []

        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        vectors = np.array(embeddings).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

        for i, chunk_id in enumerate(ids):
            self.records.append({
                "id": chunk_id,
                "text": texts[i],
                "embedding": embeddings[i],
                "document_id": metadatas[i].get("document_id", ""),
                "document_name": metadatas[i].get("document_name", ""),
                "chunk_index": metadatas[i].get("chunk_index", 0),
                "page": metadatas[i].get("page"),
            })
        self._save()

    def search(self, query_embedding: list[float], n_results: int = 3) -> dict:
        if self.index is None or self.index.ntotal == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query = np.array([query_embedding]).astype("float32")
        k = min(n_results, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        ids, docs, metas = [], [], []
        for idx in indices[0]:
            if 0 <= idx < len(self.records):
                r = self.records[idx]
                ids.append(r["id"])
                docs.append(r["text"])
                metas.append({
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "chunk_index": r["chunk_index"],
                    "page": r["page"],
                })

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances[0].tolist()]
        }

    def delete_by_document_id(self, document_id: str):
        old_count = len(self.records)
        self.records = [r for r in self.records if r["document_id"] != document_id]
        if len(self.records) == old_count:
            return
        self._rebuild_index()
        self._save()

    def _rebuild_index(self):
        if not self.records:
            self.index = None
            return
        dim = len(self.records[0]["embedding"])
        self.index = faiss.IndexFlatL2(dim)
        vectors = np.array([r["embedding"] for r in self.records]).astype("float32")
        self.index.add(vectors)

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def close(self):
        self._save()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_vector.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/db/vector.py tests/test_vector.py && git commit -m "feat: add FAISS vector store"
```

---

### Task 5: Pydantic 数据模型

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: 创建 models 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\models
touch C:\Users\LX\projects\02-research-agent\app\models\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_schemas.py
from app.models.schemas import AskRequest, AgentStep, SourceInfo


def test_ask_request_minimal():
    req = AskRequest(question="什么是 AI?")
    assert req.question == "什么是 AI?"
    assert req.conversation_id is None


def test_ask_request_with_conv():
    req = AskRequest(question="继续", conversation_id="abc123")
    assert req.conversation_id == "abc123"


def test_agent_step_with_tool():
    step = AgentStep(
        step=1,
        thought="需要检索",
        tool="search_docs",
        tool_input={"query": "test"},
        tool_result=["结果1"]
    )
    assert step.step == 1
    assert step.tool == "search_docs"


def test_agent_step_final_answer():
    step = AgentStep(
        step=2,
        thought="信息够了",
        tool=None,
        tool_input=None,
        tool_result=None
    )
    assert step.tool is None


def test_source_info():
    s = SourceInfo(type="document", name="test.pdf", content="sample text")
    assert s.type == "document"
    assert s.name == "test.pdf"
```

- [ ] **Step 3: 写 schemas.py**

```python
# app/models/schemas.py
from datetime import datetime
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(description="用户问题")
    conversation_id: str | None = Field(default=None, description="对话 ID，为空则新建")


class AgentStep(BaseModel):
    step: int = Field(description="步骤序号")
    thought: str | None = Field(description="LLM 思考过程")
    tool: str | None = Field(description="调用的工具名")
    tool_input: dict | None = Field(description="工具参数")
    tool_result: str | None = Field(description="工具返回结果")


class SourceInfo(BaseModel):
    type: str = Field(description="来源类型: document / web")
    name: str = Field(description="来源名称")
    content: str = Field(description="来源内容摘要")
    url: str | None = Field(default=None, description="URL（web 来源）")
    page: int | None = Field(default=None, description="页码（document 来源）")


class AskResponse(BaseModel):
    answer: str = Field(description="最终答案")
    steps: list[AgentStep] = Field(description="Agent 每一步记录")
    sources: list[SourceInfo] = Field(description="引用来源")
    conversation_id: str = Field(description="对话 ID")


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    created_at: datetime


class ConversationResponse(BaseModel):
    id: str
    title: str
    step_count: int
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
```

- [ ] **Step 4: 运行测试**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_schemas.py -v
```

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/models/schemas.py tests/test_schemas.py && git commit -m "feat: add Pydantic schemas"
```

---

### Task 6: 计算器工具 — 用户编写

> **这是你来写的第一个模块。** 工具函数很简单——接收参数，返回结果。关键是理解工具的定义规范。

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/calculator.py`
- Test: `tests/test_tool_calculator.py`

- [ ] **Step 1: 创建 tools 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\tools
touch C:\Users\LX\projects\02-research-agent\app\tools\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_tool_calculator.py
from app.tools.calculator import execute, get_definition


def test_simple_addition():
    result = execute("2 + 3")
    assert result["success"] is True
    assert result["result"] == 5


def test_multiplication():
    result = execute("6 * 7")
    assert result["result"] == 42


def test_complex_expression():
    result = execute("(10 + 5) * 3 / 2")
    assert result["result"] == 22.5


def test_invalid_expression_detected():
    result = execute("__import__('os').system('dir')")
    assert result["success"] is False


def test_division_by_zero():
    result = execute("1 / 0")
    assert result["success"] is False


def test_get_definition():
    definition = get_definition()
    assert definition["name"] == "calculator"
    assert "description" in definition
    assert "parameters" in definition
```

- [ ] **Step 3: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_calculator.py -v
```

Expected: 6 FAIL

- [ ] **Step 4: 实现 calculator.py**

需要实现两个函数：
- `execute(expression: str) -> dict` — 执行计算，返回结果
- `get_definition() -> dict` — 返回工具的元信息（名称、描述、参数），供 Agent Prompt 使用

```python
# app/tools/calculator.py
import re


def get_definition() -> dict:
    return {
        "name": "calculator",
        "description": "执行数学计算。当你需要进行精确的数学运算时使用此工具，例如加减乘除、百分比计算等。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2 + 3'、'(10 + 5) * 3 / 2'"
                }
            },
            "required": ["expression"]
        }
    }


def execute(expression: str) -> dict:
    # 安全校验：只允许数字、运算符、空格、括号、小数点
    allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\%]+$')
    if not allowed.match(expression):
        return {"success": False, "error": "表达式包含不允许的字符"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "result": result}
    except ZeroDivisionError:
        return {"success": False, "error": "除数不能为零"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

- [ ] **Step 5: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_calculator.py -v
```

Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/tools/calculator.py tests/test_tool_calculator.py && git commit -m "feat: add calculator tool"
```

---

### Task 7: 文档检索工具 — 用户编写

> 这个工具连接到已建的 FAISS 向量库。工具定义完后，Agent 就能搜本地文档了。

**Files:**
- Create: `app/tools/search_docs.py`
- Test: `tests/test_tool_search_docs.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_tool_search_docs.py
import pytest
from unittest.mock import MagicMock, patch
from app.tools.search_docs import execute, get_definition


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.search.return_value = {
        "ids": [["c1", "c2"]],
        "documents": [["Python 入门教程", "高级 Python 技巧"]],
        "metadatas": [[
            {"document_name": "python.pdf", "chunk_index": 0},
            {"document_name": "python.pdf", "chunk_index": 1},
        ]],
        "distances": [[0.1, 0.5]]
    }
    return mock


def test_get_definition():
    d = get_definition()
    assert d["name"] == "search_docs"
    assert "parameters" in d


def test_execute_searches(mock_vector_store):
    with patch("app.tools.search_docs._vector_store", mock_vector_store):
        result = execute("Python 基础")
    assert len(result) == 2
    assert result[0]["text"] == "Python 入门教程"


def test_execute_no_store():
    with patch("app.tools.search_docs._vector_store", None):
        result = execute("任意查询")
    assert result == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_search_docs.py -v
```

Expected: 3 FAIL

- [ ] **Step 3: 实现 search_docs.py**

```python
# app/tools/search_docs.py
from app.config import TOP_K

# 由 main.py 在启动时注入
_vector_store = None


def set_vector_store(vs):
    global _vector_store
    _vector_store = vs


def get_definition() -> dict:
    return {
        "name": "search_docs",
        "description": "在用户已上传的文档中检索相关内容。当你需要查找本地文档中的特定信息时使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                }
            },
            "required": ["query"]
        }
    }


def execute(query: str) -> list[dict]:
    if _vector_store is None:
        return []

    from app.services.llm import _llm_service
    if _llm_service is None:
        return []

    embedding = _llm_service.embed(query)
    results = _vector_store.search(embedding, n_results=TOP_K)

    chunks = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "document_name": results["metadatas"][0][i].get("document_name", ""),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "page": results["metadatas"][0][i].get("page"),
            })
    return chunks
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_search_docs.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/tools/search_docs.py tests/test_tool_search_docs.py && git commit -m "feat: add document search tool"
```

---

### Task 8: 联网搜索工具 — 用户编写

> 调用 Tavily API 进行联网搜索。需要先去 https://tavily.com 注册免费 API key（500次/月）。

**Files:**
- Create: `app/tools/web_search.py`
- Test: `tests/test_tool_web_search.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_tool_web_search.py
from unittest.mock import MagicMock, patch
from app.tools.web_search import execute, get_definition


def test_get_definition():
    d = get_definition()
    assert d["name"] == "web_search"
    assert "互联网" in d["description"]


@patch("app.tools.web_search.requests.post")
def test_execute_search(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"title": "AI 最新进展", "url": "https://example.com/ai", "content": "AI 在 2025 年..."},
            {"title": "深度学习", "url": "https://example.com/dl", "content": "深度学习是..."},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    results = execute("AI 进展")
    assert len(results) == 2
    assert results[0]["title"] == "AI 最新进展"
    assert "url" in results[0]


@patch("app.tools.web_search.requests.post")
def test_execute_api_error(mock_post):
    mock_post.side_effect = Exception("API 不可用")
    results = execute("anything")
    assert results == []
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_web_search.py -v
```

Expected: 3 FAIL

- [ ] **Step 3: 实现 web_search.py**

```python
# app/tools/web_search.py
import requests
from app.config import TAVILY_API_KEY


def get_definition() -> dict:
    return {
        "name": "web_search",
        "description": "在互联网上搜索最新信息。当你需要的信息在本地上传的文档中找不到时，使用此工具搜索互联网获取最新资料。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }


def execute(query: str) -> list[dict]:
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_key_here":
        return [{"title": "Tavily API 未配置", "url": "", "content": "请在 .env 中设置 TAVILY_API_KEY"}]

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            },
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            {
                "title": r["title"],
                "url": r["url"],
                "content": r.get("content", "")[:300],
            }
            for r in data.get("results", [])
        ]
    except Exception:
        return []
```

- [ ] **Step 4: 运行测试验证通过**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_tool_web_search.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/tools/web_search.py tests/test_tool_web_search.py && git commit -m "feat: add web search tool"
```

---

### Task 9: LLM 服务

> LLM 服务负责两件事：文本 Embedding（给检索工具用）和 Chat（含 Function Calling，给 Agent 引擎用）。

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: 创建 services 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\services
touch C:\Users\LX\projects\02-research-agent\app\services\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_llm.py
from unittest.mock import MagicMock, patch
from app.services.llm import LLMService


@patch("app.services.llm.TextEmbedding")
@patch("app.services.llm.OpenAI")
def test_embed(mock_openai, mock_embedding):
    mock_embed_model = MagicMock()
    mock_embed_model.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding.return_value = mock_embed_model

    svc = LLMService(api_key="sk-test", base_url="https://test.com")
    result = svc.embed("测试文本")

    assert isinstance(result, list)
    assert len(result) == 3


@patch("app.services.llm.TextEmbedding")
@patch("app.services.llm.OpenAI")
def test_chat(mock_openai, mock_embedding):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "你好，我是 AI 助手"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    mock_openai.return_value = mock_client

    mock_embedding.return_value = MagicMock()

    svc = LLMService(api_key="sk-test", base_url="https://test.com")
    response = svc.chat([{"role": "user", "content": "你好"}])

    assert response == "你好，我是 AI 助手"
```

- [ ] **Step 3: 写 llm.py**

```python
# app/services/llm.py
from openai import OpenAI
from fastembed import TextEmbedding

# 由 main.py 在启动时注入，供工具模块使用
_llm_service = None


class LLMService:
    def __init__(self, api_key: str, base_url: str):
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    def embed(self, text: str) -> list[float]:
        return list(self.embed_model.embed([text]))[0].tolist()

    def chat(self, messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.7) -> str:
        resp = self.chat_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content
```

- [ ] **Step 4: 运行测试**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_llm.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/services/llm.py tests/test_llm.py && git commit -m "feat: add LLM service"
```

---

### Task 10: Agent 引擎 + Prompt — 用户编写

> **这是核心模块，你来写。** Agent 引擎实现 "思考 → 执行 → 观察 → 再思考" 的循环。
> 关键点：
> 1. System Prompt 告诉 LLM 它有哪些工具、怎么用、返回什么格式
> 2. 主循环 `run()` 不断调用 LLM，解析响应，执行工具，直到 LLM 返回 `final_answer`
> 3. 每一步记录到 agent_steps 表

**Files:**
- Create: `app/services/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_agent.py
import json
import pytest
from unittest.mock import MagicMock
from app.services.agent import AgentService


@pytest.fixture
def mock_deps():
    llm = MagicMock()
    db = MagicMock()
    db.create_conversation.return_value = "conv-1"
    return llm, db


def test_agent_run_single_step(mock_deps):
    llm, db = mock_deps
    # Mock LLM 直接返回 final_answer
    llm.chat.return_value = json.dumps({
        "thought": "简单问题，直接回答",
        "action": None,
        "final_answer": "AI 是人工智能的缩写。"
    })

    agent = AgentService(llm, db)
    result = agent.run("什么是 AI?")

    assert result["answer"] == "AI 是人工智能的缩写。"
    assert result["conversation_id"] == "conv-1"
    assert len(result["steps"]) == 1


def test_agent_run_with_tool(mock_deps):
    llm, db = mock_deps
    # 第一步：LLM 决定调用 calculator
    # 第二步：LLM 返回最终答案
    call_count = [0]

    def mock_chat(messages, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps({
                "thought": "需要计算",
                "action": {"tool": "calculator", "args": {"expression": "2 + 3"}},
                "final_answer": None
            })
        else:
            return json.dumps({
                "thought": "计算结果为5",
                "action": None,
                "final_answer": "2 + 3 等于 5"
            })

    llm.chat = mock_chat
    agent = AgentService(llm, db)
    result = agent.run("2 + 3 等于多少?")

    assert "5" in result["answer"]
    assert len(result["steps"]) == 2
    assert result["steps"][0]["tool"] == "calculator"
    assert result["steps"][1]["tool"] is None


def test_agent_max_steps(mock_deps):
    llm, db = mock_deps
    # LLM 一直要求调工具，触发 MAX_STEPS 限制
    llm.chat.return_value = json.dumps({
        "thought": "还需要更多信息",
        "action": {"tool": "web_search", "args": {"query": "test"}},
        "final_answer": None
    })

    agent = AgentService(llm, db, max_steps=3)
    result = agent.run("复杂问题")

    assert len(result["steps"]) <= 3
    assert result["answer"]  # 即使超限也返回已有内容


def test_invalid_json_response(mock_deps):
    llm, db = mock_deps
    llm.chat.return_value = "这不是 JSON"

    agent = AgentService(llm, db)
    result = agent.run("问题")
    # 应对无效 JSON 的防御逻辑
    assert isinstance(result, dict)
```

- [ ] **Step 2: 运行测试验证失败**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_agent.py -v
```

Expected: 4 FAIL

- [ ] **Step 3: 实现 agent.py**

需要实现三个核心部分：

**Part A — System Prompt（你设计）：**

```python
SYSTEM_PROMPT = """你是一个智能研究助手，能够使用多种工具来回答用户的问题。

## 可用工具

1. **search_docs**: 在用户已上传的文档中检索相关内容。适用于查找本地文档中的特定信息。
   参数: {"query": "搜索关键词"}

2. **web_search**: 在互联网上搜索最新信息。适用于本地文档中没有的、需要最新资料的问题。
   参数: {"query": "搜索关键词"}

3. **calculator**: 执行数学计算。适用于需要精确数值运算的问题。
   参数: {"expression": "数学表达式，如 '2+3'、'(10+5)*3/2'"}

## 工作流程

1. 分析用户的问题
2. 决定是否需要使用工具，使用哪个工具
3. 观察工具返回的结果
4. 判断信息是否充足，是否需要继续使用工具
5. 信息充足后，给出最终答案

## 响应格式

你必须严格返回 JSON 格式，不要返回任何其他内容：

如果需要使用工具:
{
  "thought": "你当前的思考和计划",
  "action": {"tool": "工具名", "args": {"参数名": "参数值"}},
  "final_answer": null
}

如果已经可以回答:
{
  "thought": "你的总结思考",
  "action": null,
  "final_answer": "你的完整回答"
}

注意：
- 回答中引用来源时，使用 [来源X] 标注
- 如果无法找到相关信息，如实说明
- 保持回答清晰、有条理"""
```

**Part B — Agent 主循环（你实现）：**

```python
# app/services/agent.py
import json
from app.config import MAX_AGENT_STEPS
from app.tools.calculator import execute as calc_execute
from app.tools.search_docs import execute as search_execute
from app.tools.web_search import execute as web_execute


# 工具执行映射
TOOLS = {
    "calculator": calc_execute,
    "search_docs": search_execute,
    "web_search": web_execute,
}


class AgentService:
    def __init__(self, llm, db, max_steps: int = MAX_AGENT_STEPS):
        self.llm = llm
        self.db = db
        self.max_steps = max_steps

    def run(self, question: str, conversation_id: str | None = None) -> dict:
        if not conversation_id:
            conversation_id = self.db.create_conversation(question[:50])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        steps = []
        sources = []

        for step_num in range(1, self.max_steps + 1):
            # 调用 LLM
            raw = self.llm.chat(messages)
            parsed = self._parse_response(raw)

            # 记录步骤
            tool_name = None
            tool_input = None
            if parsed.get("action"):
                tool_name = parsed["action"].get("tool")
                tool_input = parsed["action"].get("args")

            self.db.add_step(
                conversation_id, step_num,
                parsed.get("thought"), tool_name,
                json.dumps(tool_input, ensure_ascii=False) if tool_input else None,
                None  # tool_result 稍后更新
            )

            steps.append({
                "step": step_num,
                "thought": parsed.get("thought"),
                "tool": tool_name,
                "tool_input": tool_input,
                "tool_result": None,
            })

            # 如果 LLM 决定返回最终答案
            if parsed.get("final_answer"):
                return {
                    "answer": parsed["final_answer"],
                    "steps": steps,
                    "sources": sources,
                    "conversation_id": conversation_id,
                }

            # 如果 LLM 要求调工具
            if parsed.get("action"):
                tool_func = TOOLS.get(tool_name)
                if tool_func:
                    try:
                        result = tool_func(**tool_input) if tool_input else tool_func()
                        result_str = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result_str = f"工具执行出错: {str(e)}"

                    # 将工具结果加入对话历史
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"工具 {tool_name} 的执行结果:\n{result_str}"
                    })
                    steps[-1]["tool_result"] = result_str
                else:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": f"工具 '{tool_name}' 不存在，可用工具: {list(TOOLS.keys())}"
                    })
            else:
                # 既没有 action 也没有 final_answer，让 LLM 继续
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "请继续：你需要调用工具还是给出最终答案？请返回 JSON 格式。"
                })

        # 达到最大步数
        return {
            "answer": "任务达到最大步数限制，以下是目前已收集的信息。请尝试简化问题或上传更多相关文档。",
            "steps": steps,
            "sources": sources,
            "conversation_id": conversation_id,
        }

    def _parse_response(self, raw: str) -> dict:
        try:
            # 尝试直接解析
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"thought": "解析响应失败", "action": None, "final_answer": raw}
```

- [ ] **Step 4: 运行测试**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_agent.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/services/agent.py tests/test_agent.py && git commit -m "feat: add agent engine with main loop"
```

---

### Task 11: 文档解析服务

**Files:**
- Create: `app/services/document.py`
- Test: `tests/test_document.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_document.py
import os
import pytest
from app.services.document import parse_file, chunk_text


def test_parse_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello World", encoding="utf-8")
    text = parse_file(str(f))
    assert text == "Hello World"


def test_parse_unsupported_extension(tmp_path):
    f = tmp_path / "test.jpg"
    f.write_text("")
    with pytest.raises(ValueError):
        parse_file(str(f))


def test_chunk_text_small():
    result = chunk_text("短文本", chunk_size=500)
    assert result == ["短文本"]


def test_chunk_text_large():
    text = "A" * 600 + "B" * 600
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert len(result) >= 2


def test_chunk_text_exact_boundary():
    text = "A" * 500
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert len(result) == 1
```

- [ ] **Step 2: 写 document.py**

```python
# app/services/document.py
import os

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def parse_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 {ALLOWED_EXTENSIONS}")

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if start >= len(text):
            break
    return chunks
```

- [ ] **Step 3: 运行测试**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_document.py -v
```

Expected: 5 PASS

- [ ] **Step 4: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/services/document.py tests/test_document.py && git commit -m "feat: add document parsing service"
```

---

### Task 12: 长期记忆模块

**Files:**
- Create: `app/memory/__init__.py`
- Create: `app/memory/memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: 创建 memory 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\memory
touch C:\Users\LX\projects\02-research-agent\app\memory\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_memory.py
import pytest
from app.memory.memory import MemoryManager


@pytest.fixture
def mem():
    from unittest.mock import MagicMock
    db = MagicMock()
    db.get_memories.return_value = [
        {"id": 1, "key": "偏好", "value": "用表格展示数据"}
    ]
    return MemoryManager(db)


def test_build_context(mem):
    ctx = mem.build_context("帮我分析销售数据")
    assert "偏好" in ctx
    assert "表格" in ctx


def test_empty_memory(mem):
    mem.db.get_memories.return_value = []
    ctx = mem.build_context("查询")
    assert ctx == ""


def test_extract_and_save_key_info(mem):
    # 从对话中提取关键信息保存
    mem.extract_and_save(
        "用户偏好",
        "用户说：我希望所有数据对比都用表格形式展示",
        "conv-1"
    )
    mem.db.save_memory.assert_called_once()
```

- [ ] **Step 3: 写 memory.py**

```python
# app/memory/memory.py


class MemoryManager:
    def __init__(self, db):
        self.db = db

    def build_context(self, user_input: str) -> str:
        """构建长期记忆上下文，注入 System Prompt"""
        memories = self.db.get_memories()
        if not memories:
            return ""

        parts = ["\n## 用户偏好 (长期记忆)"]
        for m in memories:
            parts.append(f"- {m['key']}: {m['value']}")
        return "\n".join(parts)

    def extract_and_save(self, key: str, value: str, conversation_id: str | None = None):
        """从对话中提取关键信息并保存为长期记忆"""
        self.db.save_memory(key, value, conversation_id)
```

- [ ] **Step 4: 运行测试**

```bash
cd C:\Users\LX\projects\02-research-agent && python -m pytest tests/test_memory.py -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/memory/memory.py tests/test_memory.py && git commit -m "feat: add long-term memory module"
```

---

### Task 13: API 层 — Agent 端点

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/agent.py`
- Test: `tests/test_api_agent.py`

- [ ] **Step 1: 创建 api 目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\app\api
touch C:\Users\LX\projects\02-research-agent\app\api\__init__.py
```

- [ ] **Step 2: 写测试**

```python
# tests/test_api_agent.py
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@patch("app.main.get_agent_service")
@patch("app.main.get_db")
def test_agent_run(mock_db, mock_agent):
    mock_agent.return_value.run.return_value = {
        "answer": "这是回答",
        "steps": [{"step": 1, "thought": "...", "tool": None, "tool_input": None, "tool_result": None}],
        "sources": [],
        "conversation_id": "conv-1",
    }
    mock_db.return_value = MagicMock()

    from app.main import app
    client = TestClient(app)
    resp = client.post("/api/agent/run", json={"question": "什么是 AI?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "这是回答"
    assert data["conversation_id"] == "conv-1"


def test_agent_run_empty_question():
    from app.main import app
    client = TestClient(app)
    resp = client.post("/api/agent/run", json={"question": ""})
    assert resp.status_code == 422


def test_agent_run_invalid_json():
    from app.main import app
    client = TestClient(app)
    resp = client.post("/api/agent/run", json={"wrong_field": "x"})
    assert resp.status_code == 422
```

- [ ] **Step 3: 写 agent.py 端点**

```python
# app/api/agent.py
from fastapi import APIRouter, Depends
from app.models.schemas import AskRequest, AskResponse

router = APIRouter(prefix="/api/agent", tags=["Agent"])


@router.post("/run", response_model=AskResponse)
def run_agent(req: AskRequest, agent=Depends(lambda: __import__("app.main").main.get_agent_service()),
              db=Depends(lambda: __import__("app.main").main.get_db())):
    result = agent.run(req.question, req.conversation_id)
    return result
```

- [ ] **Step 4: Commit**

*(commit 在 Task 15 完成后一起做)*

---

### Task 14: API 层 — 文档和对话端点

**Files:**
- Create: `app/api/documents.py`
- Create: `app/api/conversations.py`
- Test: `tests/test_api_documents.py`
- Test: `tests/test_api_conversations.py`

- [ ] **Step 1: 写文档端点测试**

```python
# tests/test_api_documents.py
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from io import BytesIO


@patch("app.main.get_db")
@patch("app.main.get_llm_service")
@patch("app.main.get_vector_store")
def test_list_documents(mock_vs, mock_llm, mock_db):
    mock_db.return_value.list_documents.return_value = []

    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []


@patch("app.main.get_db")
@patch("app.main.get_llm_service")
@patch("app.main.get_vector_store")
def test_delete_document_not_found(mock_vs, mock_llm, mock_db):
    mock_db.return_value.get_document.return_value = None

    from app.main import app
    client = TestClient(app)
    resp = client.delete("/api/documents/nonexistent")
    assert resp.status_code == 404
```

- [ ] **Step 2: 写对话端点测试**

```python
# tests/test_api_conversations.py
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@patch("app.main.get_db")
def test_list_conversations(mock_db):
    mock_db.return_value.list_conversations.return_value = []

    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/conversations")
    assert resp.status_code == 200


@patch("app.main.get_db")
def test_get_conversation_not_found(mock_db):
    mock_db.return_value.get_conversation.return_value = None

    from app.main import app
    client = TestClient(app)
    resp = client.get("/api/conversations/nonexistent")
    assert resp.status_code == 404
```

- [ ] **Step 3: 写 documents.py 端点**

```python
# app/api/documents.py
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schemas import DocumentResponse
from app.services.document import parse_file, chunk_text, ALLOWED_EXTENSIONS
from app.config import UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP

router = APIRouter(prefix="/api/documents", tags=["Documents"])


def _get_imports():
    import app.main
    return app.main.get_llm_service(), app.main.get_vector_store(), app.main.get_db()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}，仅支持 {ALLOWED_EXTENSIONS}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    text = parse_file(save_path)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    llm, vector_store, db = _get_imports()

    embeddings = []
    for chunk in chunks:
        embeddings.append(llm.embed(chunk))

    ids = [f"{file_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_id": file_id, "document_name": file.filename, "chunk_index": i}
        for i in range(len(chunks))
    ]
    vector_store.add(ids, chunks, embeddings, metadatas)

    doc_id = db.insert_document(
        file.filename, ext[1:], len(content), len(chunks)
    )

    return {"id": doc_id, "filename": file.filename, "chunk_count": len(chunks)}


@router.get("", response_model=list[DocumentResponse])
def list_documents():
    _, __, db = _get_imports()
    return db.list_documents()


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    _, vector_store, db = _get_imports()
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    vector_store.delete_by_document_id(doc_id)
    db.delete_document(doc_id)
    return {"message": "已删除"}
```

- [ ] **Step 4: 写 conversations.py 端点**

```python
# app/api/conversations.py
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import ConversationResponse
from app.models.schemas import AgentStep

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])


def _get_db():
    import app.main
    return app.main.get_db()


@router.get("", response_model=list[ConversationResponse])
def list_conversations():
    return _get_db().list_conversations()


@router.get("/{conv_id}")
def get_conversation(conv_id: str):
    conv = _get_db().get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "对话不存在")
    steps = _get_db().get_steps_by_conversation(conv_id)
    return {"conversation": conv, "steps": steps}


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str):
    conv = _get_db().get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "对话不存在")
    _get_db().delete_conversation(conv_id)
    return {"message": "已删除"}
```

- [ ] **Step 5: Commit**

*(commit 在 Task 15 完成后一起做)*

---

### Task 15: FastAPI 入口 (main.py)

**Files:**
- Create: `app/main.py`

- [ ] **Step 1: 写 main.py**

```python
# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, FAISS_PERSIST_DIR, SQLITE_PATH
)
from app.services.llm import LLMService, _llm_service
from app.services.agent import AgentService
from app.db.vector import VectorStore
from app.db.sqlite import Database
from app.memory.memory import MemoryManager

app = FastAPI(
    title="Research Agent - 智能研究助手",
    description="基于 AI Agent 的智能研究助手，具备文档检索、联网搜索、数学计算等能力。",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm_svc: LLMService | None = None
_vector_store: VectorStore | None = None
_db: Database | None = None
_agent: AgentService | None = None
_memory: MemoryManager | None = None


def get_llm_service() -> LLMService:
    global _llm_svc
    if _llm_svc is None:
        _llm_svc = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        # 注入到工具模块
        import app.services.llm as llm_mod
        llm_mod._llm_service = _llm_svc
    return _llm_svc


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(FAISS_PERSIST_DIR)
        import app.tools.search_docs as sd
        sd.set_vector_store(_vector_store)
    return _vector_store


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database(SQLITE_PATH)
    return _db


def get_memory_manager() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager(get_db())
    return _memory


def get_agent_service() -> AgentService:
    global _agent
    if _agent is None:
        _agent = AgentService(get_llm_service(), get_db())
    return _agent


from app.api.agent import router as agent_router
from app.api.documents import router as documents_router
from app.api.conversations import router as conversations_router

app.include_router(agent_router)
app.include_router(documents_router)
app.include_router(conversations_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: 验证服务能启动**

```bash
cd C:\Users\LX\projects\02-research-agent && HF_ENDPOINT=https://hf-mirror.com python -m uvicorn app.main:app --port 8001 &
sleep 5
curl http://localhost:8001/api/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 3: 停止服务器并运行所有测试**

```bash
kill %1 2>/dev/null
cd C:\Users\LX\projects\02-research-agent && HF_ENDPOINT=https://hf-mirror.com python -m pytest tests/ -v
```

Expected: 所有测试 PASS (~35+ 个)

- [ ] **Step 4: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add app/api/ app/main.py tests/test_api_*.py && git commit -m "feat: add API layer and app entry point"
```

---

### Task 16: Web 前端

> 使用 `ui-ux-pro-max` 生成现代科技感聊天界面。

- [ ] **Step 1: 创建静态目录**

```bash
mkdir -p C:\Users\LX\projects\02-research-agent\static
```

- [ ] **Step 2: 使用 ui-ux-pro-max 生成前端**

将生成以下文件到 `static/`:
- `index.html` — 主页面
- `style.css` — 样式
- `app.js` — 交互逻辑

页面交互要求：
- 聊天界面，输入框 + 发送按钮
- Agent 思考过程可展开/折叠（每一步显示 thought → tool → result）
- 左侧栏：对话历史列表 + 新建对话按钮
- 顶部：文档管理入口（上传、列表）
- 深色科技风主题

- [ ] **Step 3: 在 main.py 中添加静态文件挂载**

```python
# 在 app/main.py 末尾添加:
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 4: 验证前端可用**

启动服务后访问 `http://localhost:8001`，确认页面正常加载。

- [ ] **Step 5: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add static/ && git commit -m "feat: add web frontend"
```

---

### Task 17: Docker 部署

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: 写 Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV HF_ENDPOINT=https://hf-mirror.com

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/faiss && touch data/uploads/.gitkeep

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: 写 docker-compose.yml**

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8001:8000"
    env_file:
      - .env
    environment:
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - ./data:/app/data
```

- [ ] **Step 3: Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add Dockerfile docker-compose.yml && git commit -m "chore: add Docker support"
```

---

### Task 18: 最终集成验证

- [ ] **Step 1: 安装全部依赖**

```bash
cd C:\Users\LX\projects\02-research-agent && pip install -r requirements.txt -q
```

- [ ] **Step 2: 运行全部测试**

```bash
cd C:\Users\LX\projects\02-research-agent && HF_ENDPOINT=https://hf-mirror.com python -m pytest tests/ -v --tb=short
```

Expected: 全部通过 (≥ 35 个测试)

- [ ] **Step 3: 启动服务验证**

```bash
cd C:\Users\LX\projects\02-research-agent && HF_ENDPOINT=https://hf-mirror.com python -m uvicorn app.main:app --port 8001 &
sleep 5
curl http://localhost:8001/api/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: 最终 Commit**

```bash
cd C:\Users\LX\projects\02-research-agent && git add -A && git commit -m "feat: research-agent v1.0 complete"
```

---

## 用户编写任务速览

| Task | 内容 | 学到的核心 |
|------|------|-----------|
| Task 6 | 计算器工具 | 工具定义规范、沙箱安全 |
| Task 7 | 文档检索工具 | 工具与外部依赖的注入模式 |
| Task 8 | 联网搜索工具 | API 调用 + 错误兜底 |
| Task 10 | Agent 引擎 + Prompt | **核心**：Agent 循环、Prompt Engineering、JSON 解析 |
