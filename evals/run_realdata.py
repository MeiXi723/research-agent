"""真实基准数据检索评测（G2 门禁核心）：拉取后的 HotpotQA/2Wiki/MuSiQue 逐条
灌库 → 真实 bge 嵌入 → FAISS 检索 → answer-recall@k（gold 答案文本是否出现在
top-k 块中）。纯检索层验证，不依赖 LLM——离线可复算。

用法：python evals/run_realdata.py --dataset all --n 30
报告：evals/reports/report_realdata_<ts>.jsonl（逐条 + 汇总行）
"""
import argparse
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys_root = str(ROOT)
if sys_root not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_root)

DATASETS_DIR = Path(__file__).parent / "datasets"
REPORTS_DIR = Path(__file__).parent / "reports"
BENCHMARKS = ["hotpotqa", "2wiki", "musique"]


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").lower().split())


def load_items(name: str, n: int | None) -> list[dict]:
    path = DATASETS_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items[:n] if n else items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=BENCHMARKS + ["all"], default="all")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    import faiss
    import numpy as np

    from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    from app.services.document import chunk_text
    from app.services.llm import LLMService

    svc = LLMService(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    svc.embed("预热")  # 触发模型加载（本地缓存），避免计时污染
    real_model = type(svc._embed_model).__name__ != "str"
    print(f"嵌入模型: {'fastembed(bge-small-zh-v1.5)' if real_model else '⚠ 哈希降级（真实模型未加载，结果口径降级）'}")

    names = BENCHMARKS if args.dataset == "all" else [args.dataset]
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_realdata_{ts}.jsonl"

    all_records: list[dict] = []
    with open(report_path, "w", encoding="utf-8") as f:
        for name in names:
            items = load_items(name, args.n)
            if not items:
                rec = {"mode": "realdata", "dataset": name, "skipped": "数据文件缺失（拉取失败降级）"}
                all_records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            hits = 0
            latencies = []
            for i, item in enumerate(items):
                start = time.time()
                chunks = chunk_text(item["context"], chunk_size=500, overlap=50)
                if not chunks:
                    continue
                vecs = np.array([svc.embed(c) for c in chunks], dtype="float32")
                index = faiss.IndexFlatL2(vecs.shape[1])
                index.add(vecs)
                q = np.array([svc.embed(item["question"])], dtype="float32")
                k = min(args.k, index.ntotal)
                _, idx = index.search(q, k)
                top_texts = [chunks[j] for j in idx[0] if 0 <= j < len(chunks)]
                hit = any(normalize(item["answer"]) in normalize(t) for t in top_texts)
                hits += hit
                latencies.append(time.time() - start)
                rec = {
                    "mode": "realdata", "dataset": name, "idx": i, "hit": hit,
                    "question": item["question"][:80], "answer": item["answer"][:60],
                    "chunks": len(chunks), "latency_s": round(latencies[-1], 2),
                }
                all_records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total = len(items)
            summary = {
                "mode": "realdata", "dataset": name, "summary": True,
                "items": total, "recall@k": round(hits / max(total, 1), 3),
                "k": args.k,
                "avg_latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
                "embed_model": "fastembed" if real_model else "hash-fallback",
            }
            all_records.append(summary)
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            print(f"[{name}] recall@{args.k} = {hits}/{total} = {summary['recall@k']:.1%} "
                  f"(平均 {summary['avg_latency_s']}s/条)")

    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
