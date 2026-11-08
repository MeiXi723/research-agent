"""三层漏斗意图路由测试：L1 规则 → L2 语义 → L3 LLM 兜底，逐层验证命中与穿透。"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import intent_router as ir


@pytest.fixture(autouse=True)
def _fresh_samples():
    ir.reload_samples()
    yield
    ir.reload_samples()


# ── L1 规则层 ──

def test_l1_deep_keyword_wins():
    d = ir.classify_intent("帮我写一份大模型的研究报告")
    assert d["intent"] == "deep_research" and d["layer"] == 1 and d["via"] == "rule"


def test_l1_calc_expression():
    d = ir.classify_intent("3.6 / 1.2 - 0.7")
    assert d["intent"] == "tool_research" and d["layer"] == 1


def test_l1_calc_regex_no_false_positive():
    # "3个方向" 不是算式，不能被计算规则误伤
    assert ir._rule_prefilter("3个方向分别是什么") is None


def test_l1_smalltalk():
    d = ir.classify_intent("你好")
    assert d["intent"] == "chat" and d["layer"] == 1


# ── L2 语义层 ──

def _fake_embed_by_mapping(mapping: dict[str, list[float]], default: list[float]):
    return lambda text: mapping.get(text, default)


def test_l2_semantic_exact_sample_hit():
    # 样本向量与查询完全一致 → sim=1.0，margin gate 通过
    samples = ir._load_samples()
    v1 = [1.0, 0.0, 0.0]
    # 把一个样本的向量设为 v1，其余设为正交向量
    patched = [{"text": samples[0]["text"], "intent": samples[0]["intent"]}] + samples[1:]
    with patch.object(ir, "_load_samples", return_value=patched), \
         patch.object(ir, "SAMPLES_PATH", Path("nonexistent")):
        embed = _fake_embed_by_mapping(
            {patched[0]["text"]: v1, "查询问题": v1},
            [0.0, 1.0, 0.0],
        )
        d = ir.classify_intent("查询问题", embed_fn=embed)
    assert d["via"] == "semantic" and d["layer"] == 2
    assert d["intent"] == patched[0]["intent"]
    assert d["confidence"] >= ir.SEMANTIC_TOP1


def test_l2_ambiguity_falls_to_l3():
    # 所有样本向量相同 → top1==top2，margin gate 拦下 → 落 L3
    samples = [{"text": f"s{i}", "intent": "chat"} for i in range(3)]
    with patch.object(ir, "_load_samples", return_value=samples):
        def embed(t):
            return [1.0, 0.0]  # 全部同向量，无区分度
        with patch.object(ir, "_llm_classify",
                          return_value={"intent": "chat", "reason": "兜底", "via": "fallback", "layer": 3}):
            d = ir.classify_intent("模糊问题", embed_fn=embed)
    assert d["layer"] == 3


def test_l2_no_embed_fn_skips_to_llm():
    with patch.object(ir, "_llm_classify",
                      return_value={"intent": "tool_research", "reason": "", "via": "fallback", "layer": 3}):
        d = ir.classify_intent("很具体的问题", embed_fn=None)
    assert d["layer"] == 3


# ── L3 LLM 兜底 ──

def test_l3_llm_classify_success():
    with patch.object(ir.llm_module, "_llm_service") as mock_llm:
        mock_llm.chat.return_value = '前缀噪声 {"intent": "chat", "reason": "寒暄"} 后缀'
        d = ir.classify_intent("一段没有规则特征的话", embed_fn=None)
    assert d["intent"] == "chat" and d["via"] == "llm" and d["layer"] == 3


def test_l3_fallback_when_llm_raises():
    with patch.object(ir.llm_module, "_llm_service") as mock_llm:
        mock_llm.chat.side_effect = RuntimeError("api down")
        d = ir.classify_intent("很具体的问题", embed_fn=None)
    assert d["intent"] == "tool_research" and d["via"] == "fallback"


def test_l3_fallback_when_intent_out_of_range():
    with patch.object(ir.llm_module, "_llm_service") as mock_llm:
        mock_llm.chat.return_value = '{"intent": "hack", "reason": "越界"}'
        d = ir.classify_intent("怪问题", embed_fn=None)
    assert d["intent"] == "tool_research" and d["via"] == "fallback"


def test_samples_file_valid():
    data = json.loads(ir.SAMPLES_PATH.read_text(encoding="utf-8"))
    assert len(data) >= 30
    assert all(s["intent"] in ir.INTENTS for s in data)
