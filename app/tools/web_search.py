import requests

from app.config import TAVILY_API_KEY
from app.net.guard import assert_url_allowed
from app.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "在互联网上搜索最新信息。当本地文档中找不到相关信息时使用此工具。"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }

    def validate(self, query: str = "") -> str | None:
        if not query or not query.strip():
            return "搜索关键词不能为空"
        return None

    def call(self, query: str) -> list[dict]:
        if not TAVILY_API_KEY or TAVILY_API_KEY == "your-tavily-api-key":
            return []

        try:
            assert_url_allowed("https://api.tavily.com/search")  # 出口闸（P0-2）
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                content = item.get("content", "")
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": content[:300],
                })
            return results
        except Exception:
            return []
