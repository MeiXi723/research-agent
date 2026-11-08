"""语义记忆测试：余弦召回口径 + remember_smart 三分支写入判定。

向量映射要点：value 与查询文本必须**都在映射里**且指向期望向量——
映射外的文本走正交默认向量（相似度 0）。
"""
import pytest

from app.services.memory import SemanticMemory


class FakeDB:
    """memories 表最小模拟：id 自增 + update 覆盖 value/version。"""

    def __init__(self):
        self.rows = []
        self._next_id = 1

    def save_memory(self, key, value, source_conv_id=None, kind="note"):
        self.rows.append({"id": self._next_id, "key": key, "value": value,
                          "kind": kind, "version": 1, "created_at": "t0",
                          "updated_at": "t0"})
        self._next_id += 1

    def get_memories(self):
        return list(self.rows)

    def update_memory(self, memory_id, new_value):
        for r in self.rows:
            if r["id"] == memory_id:
                r["value"] = new_value
                r["version"] = r["version"] + 1
                r["updated_at"] = "t1"


class FakeLLM:
    def __init__(self, decision: str):
        self.decision = decision
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return '{"decision": "%s"}' % self.decision


def _make_memory(vectors: dict[str, list[float]]):
    """embedding 桩：文本映射到固定向量（映射外给正交默认向量，相似度 0）。"""

    def embed(text):
        return vectors.get(text, [0.0, 0.0, 1.0])

    return SemanticMemory(FakeDB(), embed_fn=embed), vectors


def test_recall_cosine_higher_is_more_similar():
    # value 与查询映射到同向量 → sim=1.0；正交查询 → sim≈0
    mem, _ = _make_memory({
        "北京今天天气晴朗": [1.0, 0.0, 0.0],
        "天气": [1.0, 0.0, 0.0],
        "完全无关的内容": [0.0, 1.0, 0.0],
    })
    mem.remember("note:1", "北京今天天气晴朗")
    assert mem.recall("天气", top_k=1)[0]["score"] == pytest.approx(1.0)
    assert mem.recall("完全无关的内容", top_k=1)[0]["score"] == pytest.approx(0.0, abs=0.01)


def test_remember_smart_high_sim_updates_existing():
    # sim=1.0 ≥ UPDATE_SIM → 更新而非新增，version+1
    mem, _ = _make_memory({
        "用户的项目截止日期是 12 月 1 日": [1.0, 0.0, 0.0],
        "用户的项目截止日期改成了 12 月 1 日之前提交": [1.0, 0.0, 0.0],  # 同向量 sim=1.0
    })
    mem.remember("fact:a", "用户的项目截止日期是 12 月 1 日")
    result = mem.remember_smart("用户的项目截止日期改成了 12 月 1 日之前提交",
                                kind="fact", llm=FakeLLM("update"))
    assert result["action"] == "updated"
    assert mem._db.rows[0]["version"] == 2
    assert len(mem._db.rows) == 1


def test_remember_smart_low_sim_creates_new():
    mem, _ = _make_memory({
        "用户的项目截止日期是 12 月 1 日": [1.0, 0.0, 0.0],
        "完全不同的新信息：用户喜欢蓝色主题": [0.0, 1.0, 0.0],
    })
    mem.remember("fact:a", "用户的项目截止日期是 12 月 1 日")
    result = mem.remember_smart("完全不同的新信息：用户喜欢蓝色主题", kind="preference")
    assert result["action"] == "created"
    assert len(mem._db.rows) == 2


def test_remember_smart_middle_band_llm_update():
    mem, _ = _make_memory({
        "项目截止日期 12 月 1 日": [0.0, 0.0, 1.0],
        "项目截止日期提前到 11 月 25 日": [0.5, 0.5, 0.71],  # cos≈0.71 中间带
    })
    mem.remember("fact:a", "项目截止日期 12 月 1 日")
    llm = FakeLLM("update")
    result = mem.remember_smart("项目截止日期提前到 11 月 25 日", kind="fact", llm=llm)
    assert result["action"] == "updated" and llm.calls == 1
    assert mem._db.rows[0]["version"] == 2


def test_remember_smart_middle_band_llm_discard():
    # 「只是当前提一下」→ discard 不入库（用户点名的核心场景）
    mem, _ = _make_memory({
        "项目截止日期 12 月 1 日": [0.0, 0.0, 1.0],
        "这个截止日期真紧张啊": [0.5, 0.5, 0.71],  # cos≈0.71 中间带
    })
    mem.remember("fact:a", "项目截止日期 12 月 1 日")
    result = mem.remember_smart("这个截止日期真紧张啊", kind="note", llm=FakeLLM("discard"))
    assert result["action"] == "discarded"
    assert len(mem._db.rows) == 1


def test_remember_smart_middle_band_llm_new():
    mem, _ = _make_memory({
        "项目截止日期 12 月 1 日": [0.0, 0.0, 1.0],
        "相关但独立的新事实：项目预算 50 万": [0.75, 0.3, 0.8],
    })
    mem.remember("fact:a", "项目截止日期 12 月 1 日")
    result = mem.remember_smart("相关但独立的新事实：项目预算 50 万", kind="fact", llm=FakeLLM("new"))
    assert result["action"] == "created"
    assert len(mem._db.rows) == 2


def test_remember_smart_empty_content_discarded():
    mem, _ = _make_memory({})
    assert mem.remember_smart("   ", kind="note")["action"] == "discarded"


def test_remember_smart_no_llm_conservative_update_in_middle_band():
    # 中间带且 LLM 不可用 → 保守更新（宁覆盖不丢信息）
    mem, _ = _make_memory({
        "项目截止日期 12 月 1 日": [0.0, 0.0, 1.0],
        "项目截止日期提前到 11 月 25 日": [0.5, 0.5, 0.71],  # cos≈0.71 中间带
    })
    mem.remember("fact:a", "项目截止日期 12 月 1 日")
    result = mem.remember_smart("项目截止日期提前到 11 月 25 日", kind="fact", llm=None)
    assert result["action"] == "updated"
