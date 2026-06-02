import os
import shutil
import pytest
from app.db.vector import VectorStore


@pytest.fixture
def vs():
    path = "./data/test_faiss"
    store = VectorStore(path)
    yield store
    store.close()
    if os.path.exists(path):
        shutil.rmtree(path)


def test_add_and_search(vs):
    ids = ["chunk-1", "chunk-2"]
    texts = ["Python 是一门编程语言", "FAISS 是向量数据库"]
    embeddings = [[0.1]*384, [0.2]*384]
    metadatas = [
        {"document_id": "doc-1", "document_name": "test.pdf", "chunk_index": 0},
        {"document_id": "doc-1", "document_name": "test.pdf", "chunk_index": 1},
    ]

    vs.add(ids, texts, embeddings, metadatas)

    result = vs.search([0.1]*384, n_results=2)
    assert len(result["ids"][0]) == 2
    assert result["documents"][0][0] == "Python 是一门编程语言"


def test_search_empty_store(vs):
    result = vs.search([0.5]*384, n_results=3)
    assert result["ids"] == [[]]


def test_delete_by_document_id(vs):
    vs.add(["c1"], ["text1"], [[0.1]*384], [{"document_id": "doc-a", "document_name": "a.pdf", "chunk_index": 0}])
    vs.add(["c2"], ["text2"], [[0.2]*384], [{"document_id": "doc-b", "document_name": "b.pdf", "chunk_index": 0}])

    vs.delete_by_document_id("doc-a")
    result = vs.search([0.1]*384, n_results=3)
    assert len(result["ids"][0]) == 1


def test_persistence(vs):
    vs.add(["c1"], ["persist test"], [[0.3]*384], [{"document_id": "doc-1", "document_name": "p.pdf", "chunk_index": 0}])
    vs.close()

    store2 = VectorStore("./data/test_faiss")
    result = store2.search([0.3]*384, n_results=1)
    assert result["documents"][0][0] == "persist test"
    store2.close()
