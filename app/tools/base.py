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
        return None

    def to_schema(self) -> dict:
        return {"type": "function", "function": self.describe()}
