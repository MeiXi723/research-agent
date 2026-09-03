import requests
from app.config import TAVILY_API_KEY


def get_definition() -> dict:
    """返回工具元信息"""
    return {
        "name": "web_search",
        "description": "在互联网上搜索最新信息。当本地文档中找不到相关信息时使用此工具。",
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


def execute(query: str) -> list[dict]:
    """
    调用 Tavily API 进行联网搜索。

    要求:
    1. 检查 TAVILY_API_KEY 是否有效（空字符串或默认值说明未配置）
    2. 用 requests.post 调用 https://api.tavily.com/search
       - json 参数: {"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 5}
       - timeout=15
    3. 解析返回的 JSON，提取 results 列表中的 title、url、content
    4. 任何异常时返回空列表（不崩溃）

    返回格式:
    [
        {
            "title": "搜索结果标题",
            "url": "https://...",
            "content": "内容摘要（截取前300字）",
        },
        ...
    ]
    """
    # TODO: 检查 API key 是否配置

    # TODO: 用 try/except 调用 Tavily API

    # TODO: 解析返回数据，组装结果

    pass
