from app.config import TOP_K
from app.tools.base import BaseTool

_vector_store = None


def set_vector_store(vs):
    global _vector_store
    _vector_store = vs


class SearchDocsTool(BaseTool):
    name = "search_docs"
    description = "在用户已上传的文档中检索相关内容。当你需要查找本地文档中的特定信息时使用此工具。"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
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

    def validate(self, query: str = "") -> str | None:
        if not query or not query.strip():
            return "搜索关键词不能为空"
        return None

    def call(self, query: str) -> list[dict]:
        if _vector_store is None:
            return []

        from app.services.llm import _llm_service

        if _llm_service is None:
            return []

        embedding = _llm_service.embed(query)
        raw_results = _vector_store.search(embedding, n_results=TOP_K)

        results = []
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        for i, text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            results.append({
                "text": text,
                "document_name": meta.get("document_name", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "page": meta.get("page"),
            })
        return results
