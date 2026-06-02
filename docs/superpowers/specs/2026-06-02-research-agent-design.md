# Research Agent 设计文档

> 独立实现的智能研究 Agent 系统，基于 Function Calling，具备多工具调用、规划-执行-反思闭环、短期+长期记忆。

**状态**: 设计阶段  
**前置项目**: 01-rag-qa (RAG 知识库问答系统)

---

## 1. 项目定位

### 1.1 一句话描述

用户上传文档或提出问题，Agent 自主决定使用哪些工具（文档检索、联网搜索、计算器），多步迭代直到完成任务，返回答案及完整的推理过程。

### 1.2 与 RAG 的核心区别

| | RAG (01-rag-qa) | Agent (02-research-agent) |
|---|---|---|
| 流程 | 检索 → 回答，一条直线 | 思考 → 行动 → 观察 → 再思考，循环迭代 |
| LLM 输出 | 自然语言 | 自然语言 + 工具调用指令 (Function Calling) |
| 工具 | 只有检索 | 检索 + 联网搜索 + 计算器，可扩展 |
| 可追溯性 | 只记录最终答案 | 每一步的思考、工具调用、结果都记录 |

---

## 2. 架构设计

### 2.1 三层结构

```
第1层 (入口):  FastAPI + 静态 Web 页面 → 接收请求，返回结果
第2层 (核心):  Agent 引擎 + 工具集 + 记忆模块 → 所有业务逻辑
第3层 (数据):  FAISS (向量) + SQLite (结构化数据) → 数据持久化
```

分层规则: 每层只与相邻层交互，不跨层调用。

### 2.2 系统组件

```
用户 (Web 页面)
    │
    ▼
┌─────────┐
│ FastAPI  │  ← API 层
└────┬────┘
     │
┌────▼──────────────────────┐
│      Agent 引擎 (核心)     │
│                           │
│  ┌──────────────────┐     │
│  │  规划模块 (Planner)│    │  ← LLM 分解任务
│  └──────┬───────────┘     │
│         │                 │
│         ▼                 │
│  ┌──────────────────┐     │
│  │  执行循环          │◄────┤  ← Agent 主循环
│  │  (Agent Loop)     │     │     "思考→行动→观察"
│  └──────┬───────────┘     │
│         │                 │
│    ┌────┴────┬────────┐   │
│    ▼         ▼        ▼   │
│ ┌─────┐ ┌──────┐ ┌─────┐ │
│ │检索  │ │搜网  │ │计算  │ │  ← 工具集 (可扩展)
│ │工具  │ │络工具 │ │器工具│ │
│ └─────┘ └──────┘ └─────┘ │
│         │                 │
│    ┌────▼────┐            │
│    │ 记忆模块 │            │  ← 短期+长期记忆
│    └─────────┘            │
└──────────────────────────┘
     │
┌────▼────┐
│  FAISS  │  ← 文档向量库
│  +SQLite│  ← 对话 + 文档元数据 + 记忆
└─────────┘
```

---

## 3. Agent 核心循环

### 3.1 主循环流程

```
用户输入问题
    │
    ▼
┌─────────────────────────────┐
│  Step 1: LLM 思考           │
│  thought: "需要先检索文档"    │
│  action: search_docs("...") │
├─────────────────────────────┤
│  执行 search_docs()         │
│  结果: 3个文档片段           │
├─────────────────────────────┤
│  Step 2: LLM 观察后思考     │
│  thought: "本地不够，联网搜"  │
│  action: web_search("...") │
├─────────────────────────────┤
│  执行 web_search()          │
│  结果: 5条搜索结果           │
├─────────────────────────────┤
│  Step 3: LLM 整合           │
│  thought: "信息充足"         │
│  action: final_answer       │
│  返回: "..."                │
└─────────────────────────────┘
    │
    ▼
返回最终答案 + steps 列表 + 来源
```

### 3.2 终止条件

- LLM 输出 `final_answer` → 正常结束
- 循环步数达到 `MAX_STEPS` (默认 10) → 强制结束，返回已有内容
- 连续 3 次同一工具调用失败 → 终止，返回错误说明

### 3.3 Function Calling 响应格式

每步 LLM 返回 JSON:

```json
{
  "thought": "分析用户问题，制定计划",
  "action": {
    "tool": "search_docs",
    "args": {"query": "数据安全"}
  }
}
```

或任务完成时:

```json
{
  "thought": "已获取足够信息",
  "action": null,
  "final_answer": "根据检索结果，文档A和文档B在数据安全方面的差异如下..."
}
```

---

## 4. 工具集

### 4.1 工具定义规范

每个工具统一接口:

```
名称: tool_name
描述: 一句话说明什么时候用
参数: {param: type, description}
返回: 结构化结果
```

### 4.2 工具清单

| 工具 | 描述 | 底层实现 | 依赖 |
|------|------|---------|------|
| `search_docs` | 在已上传文档中向量检索相关内容 | FAISS + fastembed | 无（搬到新项目） |
| `web_search` | 联网搜索，返回标题+链接+摘要 | Tavily Search API | 免费 500 次/月 |
| `calculator` | 安全计算数学表达式 | Python eval (沙箱) | 仅允许数字和运算符 |

### 4.3 工具扩展

新增工具只需:
1. 定义函数 `def tool_func(args) -> dict`
2. 注册到工具列表 `TOOLS = [search_docs, web_search, calculator]`
3. 在 Agent Prompt 中描述工具用途

---

## 5. 记忆模块

### 5.1 短期记忆

当前会话内，存 Agent 每一步的完整记录:

| 字段 | 说明 |
|------|------|
| step | 步骤序号 |
| thought | LLM 的思考过程 |
| tool | 调用的工具名 |
| tool_input | 工具参数 |
| tool_result | 工具返回结果 |

每次 LLM 决策时，将历史步骤注入 Prompt，保证上下文连续。

### 5.2 长期记忆

跨会话记忆，存入 SQLite:

| 字段 | 说明 |
|------|------|
| key | 记忆关键词 |
| value | 记忆内容 |
| source | 来源会话 ID |
| created_at | 创建时间 |

检索方式: 用户提问时，用关键词匹配历史记忆，拼入 System Prompt。

---

## 6. API 设计

### 6.1 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | Web 聊天页面 |
| `/api/documents/upload` | POST | 上传文档 (PDF/Word/TXT) |
| `/api/documents` | GET | 文档列表 |
| `/api/documents/{id}` | DELETE | 删除文档 |
| `/api/agent/run` | POST | 执行 Agent 任务 |
| `/api/conversations` | GET | 对话历史列表 |
| `/api/conversations/{id}` | GET | 对话详情 (含所有步骤) |
| `/api/conversations/{id}` | DELETE | 删除对话 |
| `/api/conversations/{id}/messages` | POST | 在同一对话中继续提问 |

### 6.2 核心请求/响应

**请求** `POST /api/agent/run`:

```json
{
  "question": "比较文档A和文档B在数据安全方面的观点差异",
  "conversation_id": null
}
```

**响应**:

```json
{
  "answer": "根据分析，文档A强调...而文档B侧重...",
  "steps": [
    {
      "step": 1,
      "thought": "需要先检索两个文档中关于数据安全的内容",
      "tool": "search_docs",
      "tool_input": {"query": "数据安全"},
      "tool_result": "[3个相关片段]"
    },
    {
      "step": 2,
      "thought": "本地文档信息不足，需要联网搜索补充",
      "tool": "web_search",
      "tool_input": {"query": "数据安全 2025"},
      "tool_result": "[5条搜索结果]"
    },
    {
      "step": 3,
      "thought": "已获取足够信息，整合回答",
      "tool": null,
      "tool_result": null
    }
  ],
  "sources": [
    {"type": "document", "name": "文档A.pdf", "chunk": "..."},
    {"type": "web", "title": "...", "url": "..."}
  ],
  "conversation_id": "uuid"
}
```

---

## 7. 数据层

### 7.1 SQLite 表结构

```sql
-- 文档
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 对话
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent 步骤记录
CREATE TABLE agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    thought TEXT,
    tool_name TEXT,
    tool_input TEXT,
    tool_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- 长期记忆
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_conversation_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 7.2 FAISS

- 复用 rag-qa 的 `IndexFlatL2` + JSON 记录方案
- 存储文档分块的 embedding 向量

---

## 8. 前端

使用 ui-ux-pro-max 生成现代科技感 Web 页面，核心交互:

- 聊天界面: 用户输入问题，流式展示 Agent 思考过程
- 步骤可视化: 展开/折叠每一步的 thought → tool → result
- 文档管理: 上传、列表、删除
- 对话历史: 左侧栏展示历史对话

---

## 9. 错误处理

| 场景 | 策略 |
|------|------|
| LLM 调用不存在工具 | 返回可用工具列表，LLM 重试 |
| 工具执行超时/失败 | 返回错误信息给 LLM，让它换方案 |
| Agent 超过最大步数 | 强制终止，返回已有内容 |
| Tavily API 额度用尽 | 提示用户，Agent 降级为仅本地检索 |
| 上传文件格式不支持 | 返回 400 + 支持的格式列表 |

---

## 10. 技术选型

| 层 | 技术 | 来源 |
|----|------|------|
| API | FastAPI | 复用 rag-qa 经验 |
| AI | DeepSeek API (Function Calling) | 已在使用 |
| 向量库 | FAISS + fastembed (BGE-small-zh) | 复用 rag-qa |
| 数据库 | SQLite | 复用 rag-qa |
| 前端 | 静态 HTML/CSS/JS | ui-ux-pro-max 生成 |
| 联网搜索 | Tavily Search API | 新增 |
| 容器 | Docker + docker-compose | 复用 rag-qa |

---

## 11. 测试策略

- Agent 循环: Mock LLM 返回预定义 JSON，验证循环解析和执行逻辑
- 工具函数: 单元测试每个工具
- API 端点: 集成测试
- 目标: ≥ 35 测试，覆盖核心路径和错误分支

---

## 12. 项目结构

```
02-research-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置读取
│   ├── api/
│   │   ├── __init__.py
│   │   ├── agent.py         # Agent 端点
│   │   ├── documents.py     # 文档端点
│   │   └── conversations.py # 对话端点
│   ├── models/
│   │   └── schemas.py       # Pydantic 模型
│   ├── services/
│   │   ├── llm.py           # LLM 服务 (含 Function Calling)
│   │   ├── agent.py         # Agent 引擎 (核心)
│   │   └── document.py      # 文档解析服务
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_docs.py   # 文档检索工具
│   │   ├── web_search.py    # 联网搜索工具
│   │   └── calculator.py    # 计算器工具
│   ├── db/
│   │   ├── sqlite.py        # SQLite 操作
│   │   └── vector.py        # FAISS 向量存储
│   └── memory/
│       └── memory.py        # 长期记忆模块
├── static/                   # Web 前端
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_api.py
│   └── ...
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-06-02-research-agent-design.md
├── .env
├── .gitignore
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 13. 与 01-rag-qa 的关系

- 独立项目，不依赖 rag-qa 代码
- FAISS 向量存储代码可借鉴
- 文档解析模块可借鉴
- API 分层结构保持一致
