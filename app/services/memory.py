# =============================================================================
# memory.py — 语义记忆（v2.1：余弦口径 + remember_smart 智能写入判定）
# =============================================================================
# sqlite 仍是事实源；embedding 只做召回索引（启动全量重建，理由见 v1 注释历史）。
#
# v2.1 变化：
# 1. 索引从 IndexFlatL2 改为归一化向量 + IndexFlatIP（内积=余弦相似度）——
#    score 从「距离越小越相关」翻转为「相似度越大越相关」，这是 remember_smart
#    阈值判定的前提（0~1 口径才能定阈值）。
# 2. remember_smart：写入判定的三分支机制（新增/更新/丢弃）——
#    sim ≥ UPDATE_SIM 直接更新；sim < NEW_SIM 直接新增；中间带 LLM 二次判定
#    （update/new/discard），「只是当前提一下」在这里被拦下。
# =============================================================================

import json
import threading
import uuid

import numpy as np

from app.db.sqlite import Database

_memory = None

# 写入判定阈值（真实 bge 余弦分布标定：同义改写 0.75~0.9，弱相关 0.5~0.7）
UPDATE_SIM = 0.80   # 高置信：同一条记忆的更新，直接覆盖
NEW_SIM = 0.62      # 低于此值：全新信息，直接新增；中间带交给 LLM 二次判定


def set_memory(m) -> None:
    global _memory
    _memory = m


def get_memory():
    return _memory


class SemanticMemory:
    """sqlite memories 表 + 余弦语义召回 + 智能写入判定。"""

    def __init__(self, db: Database, embed_fn):
        self._db = db
        self._embed = embed_fn
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def _refresh(self):
        with self._lock:
            if not self._dirty:
                return
            # 语料快照：与索引必须同源，避免并发写记忆时下标错位
            rows = list(self._db.get_memories())
            if rows:
                vecs = np.array([self._embed(r["value"]) for r in rows], dtype="float32")
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                self._matrix = vecs / norms  # 归一化 → IndexFlatIP 内积即余弦
            else:
                self._matrix = None
            self._rows = rows
            self._dirty = False

    def remember(self, key: str, value: str, source_conv_id: str | None = None,
                 kind: str = "note") -> int:
        self._db.save_memory(key, value, source_conv_id, kind)
        self._dirty = True
        return 0

    def _cosine_top(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        self._refresh()
        if self._matrix is None or not self._rows:
            return []
        qv = np.array([self._embed(query)], dtype="float32")
        qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        sims = (self._matrix @ qn.T).flatten()
        order = np.argsort(sims)[::-1][:top_k]
        return [(self._rows[int(i)], float(sims[i])) for i in order if 0 <= int(i) < len(self._rows)]

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        out = []
        for row, sim in self._cosine_top(query, top_k):
            out.append({
                "id": row["id"],
                "key": row["key"],
                "content": row["value"],
                "kind": row.get("kind", "note"),
                "version": row.get("version", 1),
                "updated_at": row.get("updated_at") or row.get("created_at"),
                "score": round(sim, 4),  # 余弦相似度，越大越相关
            })
        return out

    def remember_smart(self, content: str, kind: str = "note",
                       llm=None, source_conv_id: str | None = None) -> dict:
        """写入判定三分支：更新 / 新增 / 丢弃（用户点名机制）。

        - sim ≥ UPDATE_SIM：同一条记忆的更新 → 覆盖 value（version+1）
        - sim < NEW_SIM：全新信息 → 新增
        - 中间带：LLM 二次判定 update/new/discard；LLM 不可用时保守更新
          （宁可覆盖也不丢信息——discard 只在 LLM 明确判定「临时提及」时发生）
        """
        content = (content or "").strip()
        if not content:
            return {"action": "discarded", "reason": "空内容"}

        top = self._cosine_top(content, 1)
        sim = top[0][1] if top else 0.0
        old_row = top[0][0] if top else None

        if top and sim >= UPDATE_SIM:
            self._db.update_memory(old_row["id"], content)
            self._dirty = True
            return {"action": "updated", "memory_id": old_row["id"],
                    "similarity": round(sim, 3), "reason": f"高相似({sim:.2f})判定为更新"}

        if top and sim >= NEW_SIM:
            # 中间带：有 LLM 走二次判定；LLM 不可用 → 保守更新（宁覆盖不丢信息）
            if llm is None:
                self._db.update_memory(old_row["id"], content)
                self._dirty = True
                return {"action": "updated", "memory_id": old_row["id"],
                        "similarity": round(sim, 3),
                        "reason": f"LLM 不可用，中间带保守更新 (sim={sim:.2f})"}
            decision = self._llm_decide(llm, old_row["value"], content)
            if decision == "update":
                self._db.update_memory(old_row["id"], content)
                self._dirty = True
                return {"action": "updated", "memory_id": old_row["id"],
                        "similarity": round(sim, 3), "reason": f"LLM 判定为更新 (sim={sim:.2f})"}
            if decision == "discard":
                return {"action": "discarded", "similarity": round(sim, 3),
                        "reason": "LLM 判定为临时提及，未入长期记忆"}
            # new → 落到新增分支

        key = f"{kind}:{uuid.uuid4().hex[:8]}"
        self.remember(key, content, source_conv_id, kind)
        return {"action": "created", "key": key,
                "similarity": round(sim, 3), "reason": "新信息入库"}

    def _llm_decide(self, llm, old_value: str, new_content: str) -> str:
        try:
            prompt = (
                "判断「新信息」相对「已有记忆」应当如何处理：\n"
                "- update：新信息是已有记忆的更新/修正/补充（同一事实的新版本）\n"
                "- new：新信息是全新的独立事实\n"
                "- discard：新信息只是当前对话的临时性内容，不值得长期保存\n"
                f"已有记忆：{old_value[:300]}\n新信息：{new_content[:300]}\n"
                '只输出 JSON：{"decision": "update|new|discard"}'
            )
            raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            decision = data.get("decision", "")
            return decision if decision in ("update", "new", "discard") else "new"
        except Exception:
            return "new"  # 判定失败保守新增（宁多存不丢信息）
