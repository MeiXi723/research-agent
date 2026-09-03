from app.tools.calculator import CalculatorTool
from app.tools.web_search import WebSearchTool
from app.tools.search_docs import SearchDocsTool


def test_calculator_describe():
    tool = CalculatorTool()
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"


def test_calculator_call():
    tool = CalculatorTool()
    assert tool.call(expression="2 + 3") == {"success": True, "result": 5}


def test_calculator_validate_illegal():
    tool = CalculatorTool()
    err = tool.validate(expression="__import__('os')")
    assert err is not None


def test_calculator_validate_empty():
    tool = CalculatorTool()
    assert tool.validate() is not None


def test_web_search_validate_empty():
    tool = WebSearchTool()
    assert tool.validate() is not None


def test_search_docs_validate_empty():
    tool = SearchDocsTool()
    assert tool.validate() is not None
