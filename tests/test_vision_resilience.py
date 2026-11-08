"""视觉专线生产韧性测试：断网/超时/限流/资源错误/熔断开合，全部可控 stub。

每个故障模式对应真实上游故障场景（429 间歇限流、1113 余额、上游宕机、断网）。
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.services.vision import VisionError, VisionService

PNG_RAW = b"\x89PNG\r\n\x1a\n" + b"0" * 32  # 头部合法即可，stub 不看图


class ScriptedHandler(BaseHTTPRequestHandler):
    """行为脚本驱动的 stub：script = [("status", body) | ("timeout", sec) | ("drop",)]"""

    script: list = []
    hits = 0
    lock = threading.Lock()

    def do_POST(self):
        with ScriptedHandler.lock:
            ScriptedHandler.hits += 1
            idx = min(ScriptedHandler.hits - 1, len(ScriptedHandler.script) - 1)
            action = ScriptedHandler.script[idx]
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        kind = action[0]
        if kind == "timeout":
            time.sleep(action[1])
            return
        if kind == "drop":
            self.close_connection = True  # 直接断连（模拟断网/宕机）
            return
        status, body = action
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_factory():
    servers = []

    def make(script, **kwargs):
        ScriptedHandler.script = script
        ScriptedHandler.hits = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedHandler)  # 多线程：timeout 行为不阻塞重试请求
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        svc = VisionService(
            api_key="stub-key",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v4",
            timeout=3.0,
            retry_base_delay=0.01,  # 测试不睡真退避
            **kwargs,
        )
        return svc, server

    yield make
    for s in servers:
        s.shutdown()


OK = (200, {"choices": [{"message": {"content": "描述 OK"}}]})
QOUTA = (429, {"error": {"code": "1113", "message": "余额不足或无可用资源包,请充值。"}})
RATE = (429, {"error": {"code": "1302", "message": "并发限流"}})
ERR500 = (503, {"error": {"code": "1001", "message": "upstream down"}})
AUTH = (401, {"error": {"code": "1211", "message": "invalid key"}})


def test_rate_limit_retries_then_succeeds(stub_factory):
    svc, _ = stub_factory([RATE, RATE, OK], circuit_threshold=99)
    out = svc.describe_image(_img())
    assert out == "描述 OK"
    assert svc.status()["circuit"] == "closed"


def test_quota_error_fails_fast_no_retry(stub_factory):
    # 余额类 429 重试无意义：必须 1 次命中即失败，并带行动指引
    svc, server = stub_factory([QOUTA] * 5)
    with pytest.raises(VisionError, match="充值"):
        svc.describe_image(_img())
    assert ScriptedHandler.hits == 1  # 未重试


def test_auth_error_fails_immediately(stub_factory):
    svc, _ = stub_factory([AUTH] * 5)
    with pytest.raises(VisionError, match="API key"):
        svc.describe_image(_img())
    assert ScriptedHandler.hits == 1


def test_connection_dropped_retries_then_succeeds(stub_factory):
    # 断网/宕机（连接直接断开）→ 重试链路命中恢复后的 200
    svc, _ = stub_factory([("drop",), ("drop",), OK], circuit_threshold=99)
    out = svc.describe_image(_img())
    assert out == "描述 OK"


def test_circuit_opens_after_threshold_and_half_open_recovers(stub_factory):
    # 连续 3 次最终失败 → 熔断开 → 快速失败（不再打上游）→ 冷却后半开恢复
    svc, _ = stub_factory([ERR500] * 3 + [OK], circuit_threshold=3, circuit_cooldown=0.3)
    for _ in range(3):
        with pytest.raises(VisionError):
            svc.describe_image(_img())
    assert svc.status()["circuit"] == "open"

    hits_before = ScriptedHandler.hits
    with pytest.raises(VisionError, match="熔断"):
        svc.describe_image(_img())  # 熔断中：毫秒级快速失败
    assert ScriptedHandler.hits == hits_before  # 上游零流量

    time.sleep(0.35)  # 冷却结束 → 半开
    out = svc.describe_image(_img())
    assert out == "描述 OK"
    assert svc.status()["circuit"] == "closed"


def test_timeout_is_retryable(stub_factory):
    svc, _ = stub_factory([("timeout", 4.0), OK], circuit_threshold=99)
    svc.timeout = 0.3  # stub 睡 4s > 0.3s 超时
    out = svc.describe_image(_img())
    assert out == "描述 OK"


def _img(tmp=None) -> str:
    from pathlib import Path
    p = Path(__file__).parent / "_resilience_img.png"
    p.write_bytes(PNG_RAW)
    return str(p)
