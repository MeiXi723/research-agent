from app.config import TOP_K

# 由 main.py 启动时注入（工具模块不自己创建依赖，而是接收注入）
_vector_store = None


def set_vector_store(vs):
    """由 main.py 调用，注入向量库实例"""
    global _vector_store
    _vector_store = vs


def get_definition() -> dict:
    """返回工具元信息"""
    return {
        "name": "search_docs",
        "description": "在用户已上传的文档中检索相关内容。当你需要查找本地文档中的特定信息时使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                }
            },
            "required": ["query"]
        }
    }


def execute(query: str) -> list[dict]:
    """
    在已上传的文档中向量检索相关内容。

    要求:
    1. 检查 _vector_store 是否为 None（没有注入向量库时返回空列表）
    2. 从 app.services.llm 导入 _llm_service，用 llm_service.embed(query) 获取向量
    3. 调用 _vector_store.search(embedding, n_results=TOP_K) 检索
    4. 遍历结果，组装成列表返回

    返回格式:
    [
        {
            "text": "文档片段内容",
            "document_name": "文档名.pdf",
            "chunk_index": 0,
            "page": None,
        },
        ...
    ]
    """
    # TODO: 检查 _vector_store 是否为 None

    # TODO: 获取 LLM 服务: from app.services.llm import _llm_service

    # TODO: 调用 embed(query) 获取向量

    # TODO: 调用 _vector_store.search() 检索

    # TODO: 遍历结果组装返回

    pass
