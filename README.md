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


### LLM 提供商：火山方舟（Ark）

`.env` 三行切换（OpenAI 兼容接口，DeepSeek 官方 API 同样适用）：

```
DEEPSEEK_API_KEY=ark-xxxxxxxx
DEEPSEEK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DEEPSEEK_MODEL=deepseek-v4-1-flash-261017   # 以控制台「开通管理」实际开通的模型为准
```

**重要**：方舟的「订阅套餐」≠「模型开通」——模型 ID 直调要求先在控制台
「开通管理」页面逐个开通模型服务，否则报 `ModelNotOpen`（404）。
代码侧模型名全部由 `DEEPSEEK_MODEL` 环境变量驱动，无硬编码。
旧 DeepSeek 直连配置备份在 `.env.deepseek.bak`（不入库）。


### 多模态视觉专线（图片文档入库）

文本通道（DeepSeek）与视觉通道完全解耦——`.env` 三行接入任意 OpenAI 兼容视觉端点：

```
VISION_API_KEY=你的智谱key        # glm-4v-flash 免费
VISION_BASE_URL=https://open.bigmodel.cn/api/paas/v4
VISION_MODEL=glm-4v-flash
```

上传 `.png/.jpg/.jpeg` → 视觉模型生成可检索描述 → 入 FAISS → `/ask` 即可问到图片内容。
`VISION_API_KEY` 未配置时图片上传优雅降级（400 + 指引），文本链路不受影响。


### 安全层（v2.0 安全整改批）

| 机制 | 说明 |
|---|---|
| API 访问门禁 | `API_ACCESS_TOKEN` 配置后非本机来源强制 Bearer token（/health、/docs 放行）；留空=本机开发形态 |
| 网络出口白名单 | `NET_ALLOWED_HOSTS`（默认 DeepSeek/智谱/Tavily）——requests/httpx 层全覆盖；**MCP 子进程出口不受此闸约束**，需宿主防火墙 |
| 注入 guardrail | 工具结果进 context 前结构化分隔 + 注入模式标记（标记不删改，误杀零风险） |
| token 预算熔断 | `TOKEN_BUDGET_PER_RUN=60000`——拿到真实 usage 的运行累计超限立即中止（mock 路径不触发） |
| 审计 | X-Request-ID 响应头串联 + JSONL tracer + agent_steps + /stats + /health |

## Roadmap

- [ ] 真实 LLM 对照评测跑量（HotpotQA 等，`make compare` + `make pull-evals`）
- [ ] AI_MOCK 录制回放纳入 CI（首次真跑后 `data/ai_mock/recordings.jsonl` 入库脱敏版）
- [ ] MCP servers SSE/HTTP 传输
- [ ] 记忆写入策略（agent 自动沉淀结论，当前仅手动 remember）

## v2.0 观测层与路由漏斗（后续增补）

### 三层漏斗意图路由（`/ask` 统一入口）

```
问题 ─► L1 规则/正则（零成本） ─► L2 语义识别（热门话题样本库向量匹配） ─► L3 LLM JSON 分类 ─► 兜底 tool_research
```

- 每层命中即出漏斗，`route`/`route_via`/`layer` 字段全量可观测
- 样本库 `evals/intent_samples.json`（67 条），评测集 `evals/router_set.json` 刻意全用库外改写问法
- **覆盖契约实测达标**：`python evals/run_eval.py --mode router` → accuracy 1.0，L1+L2 覆盖 100%（目标 ≥90%）
- `force_route` 参数可绕过路由强制指定模式

### 运行级指标（AskResponse.metrics + GET /stats/{id}）

- **真实 token 用量**：捕获 OpenAI `resp.usage`（v1 是字符数冒充）；tiktoken cl100k_base 真计数驱动上下文预算
- **工具调用分析**：每工具 调用次数/成功/失败，成功率聚合（每请求 + 会话级）
- **上下文压缩事件**：压了几条消息、前后 token 数，全程可观测

### DIY 多智能体流水线（不再焊死三角色）

`pipelines/` 目录每个 JSON 声明一条流水线，任意数量角色任意编排：

```bash
GET  /api/pipelines                    # 列出全部已注册流水线
POST /api/pipelines/diy_example/run    # 跑自定义链
POST /api/research/full                # 默认三角色研究流水线（= pipelines/research.json）
```

配置字段：`engine`（research=专用图 / chain=通用线性链）、`roles[]`（name/system_prompt/tools 白名单/output_fields/route 回跳规则）。`route` 声明 `{pass_field, pass_value, revise_to, max_loops}` 即可让审核节点把不合格结果打回任意上游节点循环。`pipelines/diy_example.json` 是四角色示例（planner→searcher→analyst→reviewer 带回跳环）。

### 已知环境自愈

- fastembed 0.4.1 缓存默认在 `%TEMP%/fastembed_cache` 且国内无法在线下载——`llm.py` 启动时自动从 HF 缓存迁移模型（一次性），无需手工操作
