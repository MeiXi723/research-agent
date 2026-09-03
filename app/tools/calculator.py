import re


def get_definition() -> dict:
    """返回工具元信息，Agent 用它来决定什么时候调用这个工具"""
    return {
        "name": "calculator",
        "description": "执行数学计算。当你需要进行精确的数学运算时使用此工具。",
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


def execute(expression: str) -> dict:
    """
    安全地执行一个数学表达式并返回结果。

    要求:
    1. 用正则表达式校验 expression 只包含数字、运算符、空格、括号、小数点
    2. 校验通过后用 eval() 计算（限制 __builtins__ 为空字典防止注入）
    3. 处理除零等异常

    返回格式:
        成功: {"success": True, "result": 计算结果}
        失败: {"success": False, "error": "错误描述"}
    """
    # TODO: 用正则表达式校验 expression
    # 提示: re.compile(r'^[\d\s\+\-\*\/\(\)\.\%]+$')

    # TODO: 如果校验失败，返回 error

    # TODO: 用 eval(expression, {"__builtins__": {}}, {}) 安全计算

    # TODO: 捕获 ZeroDivisionError 和其他异常

    pass
