# research-agent — ReAct 研究智能体 v2.0

单智能体 ReAct 循环 + MCP 工具生态接入 + LangGraph 多智能体研究流水线 + 语义记忆召回，FastAPI 服务化，92 个测试全绿（含真实 stdio 子进程集成测试与 E2E 冒烟）。

## v2.0 架构

```
单智能体（/ask）
  ReAct 循环（max_steps=10, 120s 熔断）
  ├─ 内置工具: calculator / web_search / search_docs / recall_memory
  └─ MCP 动态工具: {server}__{tool}（stdio 子进程，lifespan 启停 + 孤儿检测）

多智能体（/api/research/full）
  plan → research → write → review(fact_checker 双级校验) ─┬─ finalize → END
                                                           └─ revise → write（循环）

记忆
  sqlite 事实源 + fastembed/FAISS 语义召回（recall_memory 工具）
```

| 层 | 选型 |
|---|---|
| 工具协议 | OpenAI function calling + **官方 mcp SDK 2.3**（stdio 子进程 + 独立事件循环线程） |
| 多智能体 | **LangGraph 1.2**（plan/research/write/review 条件路由） |
| 记忆 | SQLite + FAISS + fastembed（bge-small-zh-v1.5，384 维，哈希降级） |
| 服务 | FastAPI 0.115，同步 handler 走线程池；AsyncLLMService 供角色层 |
| 可靠性 | LLM 指数退避重试 + 显式 timeout；fact_checker LLM 层故障降级为规则层 |
| 测试 | 4 层：unit（mock）/ integration（真子进程）/ e2e（真 lifespan）/ eval（opt-in） |

## 快速开始

```bash
make install         # venv + 依赖（阿里云镜像）
cp .env.example .env # 最小配置：DEEPSEEK_API_KEY=sk-xxx
make run             # http://localhost:8000

# 接入 MCP 工具生态（.env 追加）：
# MCP_SERVERS=research=python mcp_servers/research_server.py
```

## 测试体系（代码绿 ≠ 跑得起来，四层证明）

```bash
make test     # L1 unit + L2 integration（含真实 stdio MCP 子进程测试），CI 必跑
make e2e      # L3 全链路冒烟：真 lifespan 启停 + 全端点，LLM mock 无网络
make compare  # L4 真实 LLM 对照评测：v1 单 agent vs v2 多 agent 同题并排
make recall   # L4 离线检索召回基线（CI 亦可跑，需 fastembed 模型缓存）
make pull-evals  # 拉取 HotpotQA/2Wiki/MuSiQue（hf-mirror，opt-in）
```

评测报告落 `evals/reports/*.jsonl`，逐行可复算（EM/F1/recall@k/步数/延迟）。

### 已覆盖的边界条件

空知识库 / max_steps 耗尽 / 120s 超时 / 工具异常与未知工具 / 坏 JSON 参数 / 工具结果 500 字符硬截断 / 上下文超限摘要降级 / MCP 子进程被杀与孤儿检测 / MCP 工具超时硬闸 / LLM 重试耗尽 / prompt 注入机制层（系统提示加固 + 探针集，攻击模式素材源自 dsh-purge redteam presets）/ fact_checker LLM 层故障降级。

## Roadmap

- [ ] 真实 LLM 对照评测跑量（HotpotQA 等，`make compare` + `make pull-evals`）
- [ ] AI_MOCK 录制回放纳入 CI（首次真跑后 `data/ai_mock/recordings.jsonl` 入库脱敏版）
- [ ] MCP servers SSE/HTTP 传输
- [ ] 记忆写入策略（agent 自动沉淀结论，当前仅手动 remember）
