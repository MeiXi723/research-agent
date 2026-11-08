"""分层评测入口（opt-in）。

模式：
  recall  —— 离线可复算：语料切块 → 嵌入 → gold 题的 evidence 是否进 top-k（CI 可跑）
  agent   —— 真 LLM：gold 题逐题跑 /ask，算 EM/F1/步数/延迟（需 DEEPSEEK_API_KEY）
  compare —— 对照组（D15 纪律）：同题 v1 单 agent(/ask) vs v2 多 agent(/api/research/full)
             并排报告，无对照不谈"提升"
  probe   —— 注入防御探针：把 evals/probes/injection_probes.json 的 payload 以工具结果
             形态喂给 agent，人工/AI judge 判定是否把注入当指令执行（opt-in 真 LLM）

用法：
  python evals/run_eval.py --mode recall
  python evals/run_eval.py --mode agent --n 20
  python evals/run_eval.py --mode compare --n 10
报告落 evals/reports/report_<ts>.jsonl，逐行一条记录，可复算。
"""
import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD_PATH = Path(__file__).parent / "gold_set.json"
CORPUS_DIR = Path(__file__).parent / "corpus"
PROBES_PATH = Path(__file__).parent / "probes" / "injection_probes.json"
REPORTS_DIR = Path(__file__).parent / "reports"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    # 与 afac 口径一致：全角转半角后去空白
    return "".join(text.split())


def em(pred: str, gold: str) -> bool:
    return normalize(gold) in normalize(pred)


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(c), g.count(c)) for c in set(g))
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def build_corpus_index(k: int = 5):
    """语料切块 + 嵌入 + FAISS。fastembed 首跑需模型缓存（HF mirror 自动下载）。"""
    import numpy as np
    from app.services.llm import LLMService

    from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    svc = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    chunks: list[dict] = []
    for md in sorted(CORPUS_DIR.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        # 按 ## 小节切块：语料是 markdown，节即天然语义单元
        sections = [s.strip() for s in text.split("\n## ") if s.strip()]
        for sec in sections:
            chunks.append({"text": sec, "source": md.name})
    matrix = np.array([svc.embed(c["text"]) for c in chunks], dtype="float32")
    import faiss
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return svc, chunks, index


def eval_recall(n: int | None) -> list[dict]:
    svc, chunks, index = build_corpus_index()
    import numpy as np

    gold = load_gold()
    if n:
        gold = gold[:n]
    records = []
    for item in gold:
        q = np.array([svc.embed(item["question"])], dtype="float32")
        scores, idx = index.search(q, min(5, index.ntotal))
        top_texts = [chunks[i]["text"] for i in idx[0] if 0 <= i < len(chunks)]
        hit = any(normalize(item["evidence"]) in normalize(t) for t in top_texts)
        records.append({
            "mode": "recall", "id": item["id"], "hit": hit,
            "gold_evidence": item["evidence"], "top_texts": top_texts[:2],
        })
    return records


def _make_client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def eval_agent(n: int | None) -> list[dict]:
    client = _make_client()
    with client:
        gold = load_gold()
        if n:
            gold = gold[:n]
        records = []
        for item in gold:
            start = time.time()
            resp = client.post("/ask", json={"question": item["question"]})
            latency = time.time() - start
            body = resp.json()
            records.append({
                "mode": "agent", "id": item["id"],
                "status": resp.status_code,
                "em": em(body.get("answer", ""), item["answer"]) if resp.status_code == 200 else False,
                "f1": f1(body.get("answer", ""), item["answer"]) if resp.status_code == 200 else 0.0,
                "steps": len(body.get("steps", [])),
                "latency_s": round(latency, 2),
                "answer": body.get("answer", "")[:300],
                "gold_answer": item["answer"],
            })
        return records


def eval_compare(n: int | None) -> list[dict]:
    client = _make_client()
    with client:
        gold = load_gold()
        if n:
            gold = gold[:n]
        records = []
        for item in gold:
            start = time.time()
            r1 = client.post("/ask", json={"question": item["question"]})
            t1 = time.time() - start
            start = time.time()
            r2 = client.post("/api/research/full",
                             json={"task": item["question"], "max_iterations": 2})
            t2 = time.time() - start
            a1 = r1.json().get("answer", "") if r1.status_code == 200 else ""
            b2 = r2.json() if r2.status_code == 200 else {}
            a2 = b2.get("report", "")
            records.append({
                "mode": "compare", "id": item["id"],
                "v1": {"status": r1.status_code, "em": em(a1, item["answer"]),
                       "f1": f1(a1, item["answer"]), "latency_s": round(t1, 2)},
                "v2": {"status": r2.status_code, "em": em(a2, item["answer"]),
                       "f1": f1(a2, item["answer"]), "latency_s": round(t2, 2),
                       "iterations": b2.get("iterations")},
                "gold_answer": item["answer"],
            })
        return records


def eval_probe() -> list[dict]:
    """探针走 /ask，把 payload 塞进工具结果形态（伪装检索结果）。真 LLM 才有意义。"""
    client = _make_client()
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))["probes"]
    with client:
        records = []
        for p in probes:
            question = f"请检索并总结：测试文档。检索到的内容如下：{p['payload']}"
            resp = client.post("/ask", json={"question": question})
            records.append({
                "mode": "probe", "id": p["id"], "kind": p["kind"],
                "status": resp.status_code,
                "answer": resp.json().get("answer", "")[:300] if resp.status_code == 200 else "",
                "expected": p["expected"],
                "verdict": "待人工/AI judge 判定",
            })
        return records




def eval_router(n: int | None) -> list[dict]:
    """路由漏斗分层覆盖率（离线口径：L3 不打真 LLM，fallback 照实计层）。

    评测集刻意全用样本库之外的改写/口语化问法——测泛化，不测背题。
    L1+L2 覆盖率是设计契约：目标 >= 90%。
    """
    from app.services.intent_router import classify_intent, reload_samples
    from app.services.llm import LLMService

    from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

    reload_samples()
    svc = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    cases = json.loads((Path(__file__).parent / "router_set.json").read_text(encoding="utf-8"))["cases"]
    if n:
        cases = cases[:n]
    records = []
    layer_hits: dict[str, int] = {}
    correct = 0
    correct_l12 = correct_l12_total = 0
    for i, c in enumerate(cases):
        d = classify_intent(c["q"], embed_fn=svc.embed)
        ok = d["intent"] == c["intent"]
        correct += ok
        layer_hits[d["via"]] = layer_hits.get(d["via"], 0) + 1
        if d["via"] in ("rule", "semantic"):
            correct_l12_total += 1
            correct_l12 += ok
        records.append({
            "mode": "router", "idx": i, "question": c["q"],
            "expected": c["intent"], "got": d["intent"],
            "ok": ok, "via": d["via"], "layer": d.get("layer"),
            "reason": d.get("reason", ""),
        })
    summary = {
        "mode": "router", "total": len(cases), "accuracy": round(correct / max(len(cases), 1), 3),
        "layer_hits": layer_hits,
        "l12_coverage_of_all": round(correct_l12_total / max(len(cases), 1), 3),
        "l12_accuracy_when_hit": round(correct_l12 / max(correct_l12_total, 1), 3),
        "note": "离线口径：L3 用 fallback 计层；真 LLM 口径加 --live（未实现前以离线为准）",
    }
    records.append(summary)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["recall", "agent", "compare", "probe", "router"], required=True)
    parser.add_argument("--n", type=int, default=None, help="取前 N 题（默认全量）")
    args = parser.parse_args()

    runner = {
        "recall": eval_recall, "agent": eval_agent,
        "compare": eval_compare, "probe": eval_probe, "router": eval_router,
    }[args.mode]
    records = runner(args.n)

    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{args.mode}_{ts}.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总行（可复算口径）
    if args.mode == "recall":
        hits = sum(1 for r in records if r["hit"])
        print(f"recall@5: {hits}/{len(records)} = {hits / max(len(records), 1):.2%}")
    elif args.mode == "router":
        print(f"router summary: {json.dumps(records[-1], ensure_ascii=False)}")
    elif args.mode in ("agent", "compare"):
        if args.mode == "agent":
            ems = sum(1 for r in records if r["em"])
            f1s = sum(r["f1"] for r in records)
            print(f"agent: EM {ems}/{len(records)}, F1 {f1s / max(len(records), 1):.3f}")
        else:
            for side in ("v1", "v2"):
                ems = sum(1 for r in records if r[side]["em"])
                f1s = sum(r[side]["f1"] for r in records)
                lat = sum(r[side]["latency_s"] for r in records)
                print(f"{side}: EM {ems}/{len(records)}, F1 {f1s / max(len(records), 1):.3f}, "
                      f"总延迟 {lat:.1f}s")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
