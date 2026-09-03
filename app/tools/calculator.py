import re

from app.tools.base import BaseTool

_EXPRESSION_PATTERN = re.compile(r'^[\d\s\+\-\*\/\(\)\.\%]+$')


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行数学计算。当你需要进行精确的数学运算时使用此工具。"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '2 + 3'、'(10 + 5) * 3 / 2'"
                    }
                },
                "required": ["expression"]
            }
        }

    def validate(self, expression: str = "") -> str | None:
        if not expression or not expression.strip():
            return "表达式不能为空"
        if not _EXPRESSION_PATTERN.match(expression):
            return f"表达式包含非法字符: {expression}"
        return None

    def call(self, expression: str) -> dict:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"success": True, "result": result}
        except ZeroDivisionError:
            return {"success": False, "error": "除零错误"}
        except Exception as e:
            return {"success": False, "error": f"计算错误: {e}"}
