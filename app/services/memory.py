# =============================================================================
# memory.py — 语义记忆（v2.0）
# =============================================================================
# v1 的 memories 表是纯 key-value，只能按 key 精确取——「上次查过的结论」
# 这类记忆没法按语义找回。本层给记忆加向量召回：
# sqlite 仍是事实源（key/content/元信息），embedding 只做召回索引。
#
# 为什么启动时全量重建而不持久化 embedding：记忆条目是小集合（百级），
# 重建 = 重新 embed 一遍，毫秒级；持久化反而引入「embedding 与 sqlite
# 漂移」的第二事实源问题（14 号项目 kb_index.json 的教训）。
# =============================================================================

import threading

import faiss
import numpy as np

from app.db.sqlite import Database

_memory = None


def set_memory(m) -> None:
    global _memory
    _memory = m


def get_memory():
    return _memory


class SemanticMemory:
    """sqlite memories 表 + FAISS 语义召回。dirty 后惰性重建（快照语义，见 _refresh）。"""

    def __init__(self, db: Database, embed_fn):
        self._db = db
        self._embed = embed_fn
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._index: faiss.IndexFlatL2 | None = None
        self._dirty = True

    def _refresh(self):
        with self._lock:
            if not self._dirty:
                return
            # 语料快照：与索引必须同源，避免并发写记忆时下标错位
            rows = list(self._db.get_memories())
            if rows:
                vecs = np.array([self._embed(r["value"]) for r in rows], dtype="float32")
                index = faiss.IndexFlatL2(vecs.shape[1])
                index.add(vecs)
            else:
                index = None
            self._rows = rows
            self._index = index
            self._dirty = False

    def remember(self, key: str, value: str, source_conv_id: str | None = None) -> int:
        self._db.save_memory(key, value, source_conv_id)
        self._dirty = True
        return self._db.get_memories()[0]["id"] if self._db.get_memories() else -1

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        self._refresh()
        if self._index is None or not self._rows:
            return []
        q = np.array([self._embed(query)], dtype="float32")
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q, k)
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._rows):
                continue
            row = self._rows[idx]
            out.append({
                "key": row["key"],
                "content": row["value"],
                "created_at": row["created_at"],
                "score": round(float(dist), 4),  # L2 距离，越小越相关
            })
        return out
