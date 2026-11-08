"""多模态视觉专线测试：unit（mock）+ integration（本地 OpenAI 兼容 stub 真实 HTTP）。"""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.services.vision import VisionError, VisionService, VisionUnavailableError

pytestmark = pytest.mark.integration

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

STUB_RESPONSE = {
    "choices": [{"message": {"content": "图中是一张系统参数表：最大步数 99 步，专线端口 4242。"}}],
}


# ── unit：未配置 / 调用失败路径 ──

def test_unconfigured_raises_unavailable(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_1PX)
    svc = VisionService(api_key="")  # 未配 key
    with pytest.raises(VisionUnavailableError):
        svc.describe_image(str(img))


def test_http_error_wrapped(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_1PX)

    class FakeResp:
        status_code = 429
        text = "rate limited"
    monkeypatch.setattr("app.services.vision.requests.post",
                        lambda *a, **k: FakeResp())
    svc = VisionService(api_key="k")
    with pytest.raises(VisionError, match="429"):
        svc.describe_image(str(img))


def test_bad_structure_wrapped(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_1PX)

    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": []}
    monkeypatch.setattr("app.services.vision.requests.post",
                        lambda *a, **k: FakeResp())
    svc = VisionService(api_key="k")
    with pytest.raises(VisionError, match="响应结构异常"):
        svc.describe_image(str(img))


def test_unsupported_ext(tmp_path):
    img = tmp_path / "a.gif"
    img.write_bytes(b"GIF89a")
    svc = VisionService(api_key="k")
    with pytest.raises(VisionError, match="不支持的图片格式"):
        svc.describe_image(str(img))


# ── integration：本地 stub 真实 HTTP 全链路 ──

class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.loads(body)
        # 校验 OpenAI 兼容视觉消息形状
        content = payload["messages"][0]["content"]
        assert any(c.get("type") == "image_url" for c in content)
        assert any(c["image_url"]["url"].startswith("data:image/png;base64,") for c in content)
        resp = json.dumps(STUB_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):  # 静音
        pass


@pytest.fixture
def vision_app():
    """起本地 stub 视觉端点 + 配置 VISION 指向它 + 起真实 app（临时目录）。"""
    import os
    import tempfile

    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    tmp = tempfile.mkdtemp(prefix="vision_eval_")
    ctx = [
        patch("app.main.VISION_API_KEY", "stub-key"),
        patch("app.main.VISION_BASE_URL", f"http://127.0.0.1:{port}/v4"),
        patch("app.main.VISION_MODEL", "glm-4v-flash-stub"),
        patch("app.main.SQLITE_PATH", os.path.join(tmp, "eval.db")),
        patch("app.main.UPLOAD_DIR", os.path.join(tmp, "uploads")),
        patch("app.main.FAISS_PERSIST_DIR", os.path.join(tmp, "faiss")),
        patch("app.main.TRACE_DIR", os.path.join(tmp, "traces")),
    ]
    for p_ in ctx:
        p_.start()

    for d in ("uploads", "faiss", "traces"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    from app.main import app
    # 必须 with：lifespan（db/faiss/vision 单例）只在上下文内执行
    with TestClient(app) as client:
        yield client
    for p_ in ctx:
        p_.stop()
    server.shutdown()


def test_image_upload_to_retrieval_full_chain(vision_app):
    # 1. 上传图片 → 视觉专线描述 → 入库
    up = vision_app.post(
        "/documents/upload",
        files={"file": ("table.png", PNG_1PX, "image/png")},
    )
    assert up.status_code == 200, up.text
    doc = up.json()
    assert doc["file_type"] == "png" and doc["chunk_count"] == 1

    # 2. 图片描述可被语义检索命中（stub 描述含「4242」与「99 步」）
    ask = vision_app.post(
        "/ask",
        json={"question": "专线端口是多少？", "force_route": "tool_research"},
    )
    assert ask.status_code == 200
    sources = ask.json()["sources"]
    assert any("4242" in (s.get("content") or "") for s in sources), \
        f"图片描述未被检索召回: {sources}"


def test_image_upload_without_vision_key_degrades_gracefully(tmp_path):
    # 专线未配置 → 400 + 明确指引（不 500、不影响文本链路）
    import os

    with patch("app.main.VISION_API_KEY", ""), \
         patch("app.main.SQLITE_PATH", os.path.join(tmp_path, "t.db")), \
         patch("app.main.UPLOAD_DIR", os.path.join(tmp_path, "up")), \
         patch("app.main.FAISS_PERSIST_DIR", os.path.join(tmp_path, "faiss")), \
         patch("app.main.TRACE_DIR", os.path.join(tmp_path, "tr")):
        from app.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/documents/upload",
                files={"file": ("n.png", PNG_1PX, "image/png")},
            )
            assert resp.status_code == 400
            assert "视觉专线未配置" in resp.json()["detail"]
