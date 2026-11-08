# PRODUCTION_GOAL — research-agent 生产级落地目标（2026-11-08 立项）

> 定位：从「功能完整的 v2.0」推进到「有真实数据验证、可量化验收的生产落地」。
> 每项门禁都有复算口径（评测脚本 + JSONL 报告），禁止引用记忆值。

## G1 功能与质量门禁（已达标 ✅）

| 门禁 | 目标 | 实测 | 复算方式 |
|---|---|---|---|
| 测试规模 | 全绿 + 关键路径覆盖 | 135（89 单测/集成 + 3 e2e + 32 DIY + …）全绿 | `make test && make e2e` |
| CI 门禁 | GitHub Actions 绿 | ✅（4a53456） | gh run list |
| 意图路由漏斗 | L1+L2 覆盖 ≥90% 且准确率 ≥95% | **100% / 100%** | `run_eval.py --mode router` |
| 本地金标检索 | recall@5 ≥0.8 | **26/30 = 86.7%** | `run_eval.py --mode recall` |
| 安全 | 路径穿越/注入探针/超长截断 | 机制层测试全绿 + 10 条注入探针 | test_agent_probes |
| 工程化 | CI+README+Makefile+healthcheck | ✅ | 目检 |

## G2 真实数据验证门禁（本轮推进 🔥）

| 门禁 | 目标 | 状态 |
|---|---|---|
| 真实基准数据入库 | ✅ HotpotQA 30（段落+supporting_facts gold）/2Wiki 30/MuSiQue 30（pos_doc 段落）经镜像+rows API 双通道拉取 | ✅ |
| 真实检索评测 | answer-recall@5 ≥0.75（真实 bge 嵌入，段落感知切块） | ✅ **hotpotqa 93.3% / 2wiki 83.3% / musique 100%（均值 92.2%）**，报告 report_realdata_20261108_130511 |
| 真实 LLM 端到端 | ✅ **EM 6/10（字面）/ judge 8/10（LLM-as-judge 语义口径）**；失败项分析：1 真检索失败（sys08）+ 1 复合格（sys04），其余为字面口径误伤 | 报告 report_agent_20261108_135722 |
| 成本计量 | ✅ **10 题合计 27,133 tokens（均值 2,713/题）**，远低于 15k/题预算门禁；token 明细逐题入库 |
| v1 vs v2 对照 | ✅ 单跳事实题：v1 18.1s vs v2 329.9s（18 倍延迟）且 EM 持平（4/5）——**证实路由层是成本/延迟守门员**：单跳必须走 tool_research，多 agent 只服务深度任务（多跳综合报告场景另行评测） |
| 并发稳定性 | 5 并发 /ask 无串话、无状态污染、错误率 0 | ✅ 并发冒烟全绿；**抓到并修复 SQLite 共享连接并发写竞态**（insert-then-read 返回 None）+ agent 指标 sink 覆盖路由字段两缺陷 |

## G3 生产交付形态门禁（下一步）

- [ ] Docker 镜像构建验证（`docker compose build` + healthcheck 通过）
- [ ] AI_MOCK 录制集入库（真 key 首跑后录制 → CI 全离线冒烟）
- [ ] 评测报告归档纪律：每次 run_eval 报告留存 evals/reports/ 并在 README 记录最新数字
- [ ] 冷启动预热（embedding/MCP server 启动即加载，消灭首请求超时）

## 阻塞与解法

1. **LLM 提供商切换完成，等控制台开通模型**：`.env` 已切火山方舟（key 认证实测通过，`override=True` 修复旧 shell 环境变量压配置的缺陷），模型 ID 直调报 `ModelNotOpen`——需在方舟控制台「开通管理」开通目标模型（如 deepseek-v4-1-flash-261017）后即闭环 G2 的 LLM 三项（agent EM/F1、v1/v2 对照、成本报告）。检索层/路由层/并发层的真实数据验证已在无 LLM 依赖下完成。
   换有效 key 到 `.env` 后依次跑：
   `python evals/run_eval.py --mode agent --n 10` → `--mode compare --n 5` → 并发真跑。
2. 数据集候选 id 拉取失败不静默：pull_datasets 逐候选实测，失败降级并在 stderr 声明。
