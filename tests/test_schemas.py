from app.models.schemas import AskRequest, AgentStep, SourceInfo


def test_ask_request_minimal():
    req = AskRequest(question="什么是 AI?")
    assert req.question == "什么是 AI?"
    assert req.conversation_id is None


def test_ask_request_with_conv():
    req = AskRequest(question="继续", conversation_id="abc123")
    assert req.conversation_id == "abc123"


def test_agent_step_with_tool():
    step = AgentStep(
        step=1,
        thought="需要检索",
        tool="search_docs",
        tool_input={"query": "test"},
        tool_result='["结果1"]'
    )
    assert step.step == 1
    assert step.tool == "search_docs"


def test_agent_step_final_answer():
    step = AgentStep(
        step=2,
        thought="信息够了",
        tool=None,
        tool_input=None,
        tool_result=None
    )
    assert step.tool is None


def test_source_info():
    s = SourceInfo(type="document", name="test.pdf", content="sample text")
    assert s.type == "document"
    assert s.name == "test.pdf"
