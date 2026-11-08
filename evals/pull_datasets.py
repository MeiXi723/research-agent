"""从 hf-mirror 拉取多跳 QA 评测基准（opt-in，需先 pip install datasets）。

三套均为"自带上下文段落"的多跳 QA——不依赖外网搜索，灌入本地 FAISS 后即可评测。
具体 HF dataset id 以拉取时实测为准：候选 id 列表逐个尝试，字段校验通过才算可用；
全部失败则打印结论并退出（不静默假绿）。

用法：HF_ENDPOINT=https://hf-mirror.com python evals/pull_datasets.py --n 50
产出：evals/datasets/<name>.jsonl，每行 {"question","answer","context"}
"""
import argparse
import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent / "datasets"

# 候选 id 按优先级排列（HF 上镜像/转发仓众多，拉取时以第一个字段校验通过的为准）
BENCHMARKS = {
    "hotpotqa": {
        "candidates": ["hotpotqa/hotpot_qa"],
        "config": "distractor",
    },
    "2wiki": {
        "candidates": [
            "framolfese/2WikiMultihopQA",
            "voidful/2WikiMultihopQA",
            "xanhho/2WikiMultihopQA",
        ],
        "config": None,
    },
    "musique": {
        "candidates": ["dgslibisey/MuSiQue"],
        "config": None,
    },
}


def extract_fields(row: dict) -> dict | None:
    q = row.get("question") or row.get("query")
    a = row.get("answer")
    if not q or not a:
        return None
    # context 形态兼容：list[list]（hotpot/musique）或 dict of lists（2wiki 变体）
    ctx = row.get("context") or row.get("paragraphs")
    if isinstance(ctx, dict):
        texts = [t for seg in ctx.get("sentences", []) for t in seg]
    elif isinstance(ctx, list):
        texts = []
        for seg in ctx:
            if isinstance(seg, list):
                texts.extend(t for t in seg if isinstance(t, str))
            elif isinstance(seg, str):
                texts.append(seg)
    else:
        texts = []
    return {"question": q, "answer": str(a), "context": "\n".join(texts)[:4000]}


def try_pull(name: str, spec: dict, n: int) -> bool:
    from datasets import load_dataset

    for repo in spec["candidates"]:
        try:
            print(f"[{name}] 尝试 {repo} ...", file=sys.stderr)
            ds = load_dataset(repo, spec["config"], split="validation" if name == "hotpotqa" else "train")
            out = OUT_DIR / f"{name}.jsonl"
            count = 0
            with open(out, "w", encoding="utf-8") as f:
                for row in ds:
                    rec = extract_fields(row)
                    if rec:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= n:
                            break
            if count >= min(n, 10):  # 至少拉到像样的量才算可用
                print(f"[{name}] OK: {repo} → {out}（{count} 条）", file=sys.stderr)
                return True
            print(f"[{name}] {repo} 字段校验/数量不足（{count} 条），试下一个候选", file=sys.stderr)
        except Exception as e:
            print(f"[{name}] {repo} 失败: {type(e).__name__}: {e}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="每套基准抽取条数")
    parser.add_argument("--only", type=str, default=None, help="只拉某套（hotpotqa/2wiki/musique）")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    OUT_DIR.mkdir(exist_ok=True)

    names = [args.only] if args.only else list(BENCHMARKS)
    ok, failed = [], []
    for name in names:
        if try_pull(name, BENCHMARKS[name], args.n):
            ok.append(name)
        else:
            failed.append(name)
    print(f"完成: 可用={ok} 失败={failed}（失败的按方案降级，评测只用可用集）", file=sys.stderr)


if __name__ == "__main__":
    main()
