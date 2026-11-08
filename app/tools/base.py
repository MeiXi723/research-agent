from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def describe(self) -> dict:
        ...

    @abstractmethod
    def call(self, **kwargs):
        ...

    def validate(self, **kwargs) -> str | None:
        """语义校验钩子（各工具自行实现），返回错误串或 None。"""
        return None

    def validate_typed(self, kwargs: dict) -> str | None:
        """对照 describe() 的 JSON Schema 做结构校验（defect-audit P2-4）：
        必填缺失、未知参数、基本类型不匹配——错误信息面向 LLM 自纠。
        子类的 validate() 继续负责语义校验，两者叠加。"""
        schema = (self.describe() or {}).get("parameters", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [r for r in required if kwargs.get(r) in (None, "")]
        if missing:
            return f"缺少必填参数: {missing}（该工具要求: {required}）"

        unknown = [k for k in kwargs if k not in props]
        if unknown:
            return f"未知参数: {unknown}，该工具只接受: {list(props)}"

        type_map = {"string": str, "integer": int,
                    "number": (int, float), "boolean": bool}
        for k, v in kwargs.items():
            want = props.get(k, {}).get("type")
            expected = type_map.get(want)
            if expected is None:
                continue
            # bool 是 int 的子类——声明 integer 时收到 bool 要报错
            if isinstance(v, bool) and want != "boolean":
                return f"参数 {k} 类型应为 {want}，收到 boolean"
            if not isinstance(v, expected):
                return f"参数 {k} 类型应为 {want}，收到 {type(v).__name__}"
        return None

    def to_schema(self) -> dict:
        return {"type": "function", "function": self.describe()}
