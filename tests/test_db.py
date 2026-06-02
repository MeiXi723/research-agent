import os
import pytest
from app.db.sqlite import Database


@pytest.fixture
def db():
    path = "./data/test_research.db"
    database = Database(path)
    yield database
    database.close()
    if os.path.exists(path):
        os.remove(path)
    for ext in ["-shm", "-wal"]:
        if os.path.exists(path + ext):
            os.remove(path + ext)


def test_insert_and_get_document(db):
    doc_id = db.insert_document("test.pdf", "pdf", 1024, 5)
    assert doc_id

    doc = db.get_document(doc_id)
    assert doc["filename"] == "test.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["chunk_count"] == 5


def test_list_documents(db):
    db.insert_document("a.pdf", "pdf", 100, 2)
    db.insert_document("b.txt", "txt", 200, 1)
    docs = db.list_documents()
    assert len(docs) == 2


def test_delete_document(db):
    doc_id = db.insert_document("x.pdf", "pdf", 50, 1)
    db.delete_document(doc_id)
    assert db.get_document(doc_id) is None


def test_create_and_get_conversation(db):
    conv_id = db.create_conversation("测试对话")
    assert conv_id

    conv = db.get_conversation(conv_id)
    assert conv["title"] == "测试对话"


def test_list_conversations(db):
    db.create_conversation("对话1")
    db.create_conversation("对话2")
    convs = db.list_conversations()
    assert len(convs) == 2


def test_delete_conversation(db):
    conv_id = db.create_conversation("待删除")
    db.delete_conversation(conv_id)
    assert db.get_conversation(conv_id) is None


def test_add_and_get_agent_steps(db):
    conv_id = db.create_conversation("测试")
    db.add_step(conv_id, 1, "搜索一下", "search_docs", '{"query":"test"}', '["结果1"]')
    db.add_step(conv_id, 2, "信息够了", None, None, None)

    steps = db.get_steps_by_conversation(conv_id)
    assert len(steps) == 2
    assert steps[0]["thought"] == "搜索一下"
    assert steps[1]["tool_name"] is None


def test_memories_crud(db):
    db.save_memory("用户偏好", "喜欢表格", "conv-1")
    memories = db.get_memories()
    assert len(memories) == 1
    assert memories[0]["key"] == "用户偏好"

    mid = memories[0]["id"]
    db.delete_memory(mid)
    assert len(db.get_memories()) == 0
