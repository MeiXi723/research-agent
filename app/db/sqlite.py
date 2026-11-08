import json
import sqlite3
import uuid
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _uid():
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        # 单共享连接 + check_same_thread=False：线程池并发写时游标交错、commit
        # 互相吞——并发冒烟实测「插入后立查返回 None」。写路径全走这把锁。
        import threading
        self._write_lock = threading.RLock()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._write_lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                thought TEXT,
                tool_name TEXT,
                tool_input TEXT,
                tool_result TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_conversation_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_messages (
                conversation_id TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)

    def close(self):
        self.conn.close()

    # ── documents ──

    def insert_document(self, filename: str, file_type: str, file_size: int, chunk_count: int) -> str:
        doc_id = _uid()
        self._write(
            "INSERT INTO documents (id, filename, file_type, file_size, chunk_count, created_at) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, file_type, file_size, chunk_count, _now())
        )
        return doc_id

    def get_document(self, doc_id: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def delete_document(self, doc_id: str):
        self._write("DELETE FROM documents WHERE id=?", (doc_id,))

    # ── conversations ──

    def create_conversation(self, title: str = "") -> str:
        conv_id = _uid()
        self._write(
            "INSERT INTO conversations (id, title, created_at) VALUES (?,?,?)",
            (conv_id, title, _now())
        )
        return conv_id

    def get_conversation(self, conv_id: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,))
        row = cur.fetchone()
        if not row:
            return None
        conv = dict(row)
        cur = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM agent_steps WHERE conversation_id=?", (conv_id,)
        )
        conv["step_count"] = cur.fetchone()["cnt"]
        return conv

    def list_conversations(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM conversations ORDER BY created_at DESC")
        convs = [dict(r) for r in cur.fetchall()]
        for c in convs:
            cur = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM agent_steps WHERE conversation_id=?", (c["id"],)
            )
            c["step_count"] = cur.fetchone()["cnt"]
        return convs

    def delete_conversation(self, conv_id: str):
        self._write("DELETE FROM agent_steps WHERE conversation_id=?", (conv_id,))
        self._write("DELETE FROM conversations WHERE id=?", (conv_id,))

    # ── agent steps ──

    def add_step(self, conv_id: str, step_number: int, thought: str | None,
                 tool_name: str | None, tool_input: str | None, tool_result: str | None):
        self._write(
            "INSERT INTO agent_steps (conversation_id, step_number, thought, "
            "tool_name, tool_input, tool_result, created_at) VALUES (?,?,?,?,?,?,?)",
            (conv_id, step_number, thought, tool_name, tool_input, tool_result, _now())
        )

    def get_steps_by_conversation(self, conv_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM agent_steps WHERE conversation_id=? ORDER BY step_number ASC",
            (conv_id,)
        )
        return [dict(r) for r in cur.fetchall()]

    # ── memories ──

    def save_memory(self, key: str, value: str, source_conv_id: str | None = None):
        self._write(
            "INSERT INTO memories (key, value, source_conversation_id, created_at) VALUES (?,?,?,?)",
            (key, value, source_conv_id, _now())
        )

    def get_memories(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM memories ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def delete_memory(self, memory_id: int):
        self._write("DELETE FROM memories WHERE id=?", (memory_id,))

    # ── conversation messages ──

    def save_conversation_messages(self, conv_id: str, messages: list[dict]):
        self._write(
            "INSERT INTO conversation_messages (conversation_id, messages_json, updated_at) "
            "VALUES (?,?,?) "
            "ON CONFLICT(conversation_id) DO UPDATE SET "
            "messages_json=excluded.messages_json, updated_at=excluded.updated_at",
            (conv_id, json.dumps(messages, ensure_ascii=False), _now())
        )

    def get_conversation_messages(self, conv_id: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT messages_json FROM conversation_messages WHERE conversation_id=?", (conv_id,)
        )
        row = cur.fetchone()
        if not row:
            return []
        return json.loads(row["messages_json"])
