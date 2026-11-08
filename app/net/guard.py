# =============================================================================
# guard.py — 网络出口白名单（安全评估 P0-2）
# =============================================================================
# 覆盖边界（如实声明）：本闸只约束 Python 进程内的 requests/httpx 直连出口
# （llm/vision/web_search）。MCP stdio 子进程的网络不受此闸约束——子进程出口
# 管控需宿主防火墙（iptables/安全组），文档化于 README。
#
# D21/D27 纪律：白名单是闸类机制——测试必须双向覆盖：
# ①闸拦得住（非白名单 host 抛 NetworkPolicyError）
# ②登记后必绿（新合法 host 加白后通过，误拦方向）
# =============================================================================

from urllib.parse import urlparse

from app.config import NET_ALLOWED_HOSTS


class NetworkPolicyError(RuntimeError):
    """出口违反网络白名单策略。携带 host 与放行指引。"""


def allowed_hosts() -> list[str]:
    return list(NET_ALLOWED_HOSTS)


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def assert_host_allowed(url: str) -> None:
    """请求发出前的出口闸。host 不在白名单 → NetworkPolicyError。

    回环地址（127.0.0.1/localhost/::1）永远放行——本机 stub/测试端点与本地
    调试属合理出口（P-150 误拦方向：出口闸上线首日即误拦 127.0.0.1 测试端点，
    由此固化回环放行）。
    """
    host = host_of(url)
    if not host:
        raise NetworkPolicyError(f"无法解析出口 host: {url[:120]}")
    if host in LOOPBACK_HOSTS:
        return
    allowed = [h.lower() for h in allowed_hosts()]
    if host not in allowed:
        raise NetworkPolicyError(
            f"网络出口被白名单拦截: {host} 不在 NET_ALLOWED_HOSTS。"
            f"如为合法端点，将 {host} 加入 .env 的 NET_ALLOWED_HOSTS 后重启。"
        )


def assert_url_allowed(url: str) -> None:
    """同 assert_host_allowed，别名（语义清晰）。"""
    assert_host_allowed(url)
