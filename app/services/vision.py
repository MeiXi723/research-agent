# =============================================================================
# vision.py — 多模态视觉专线（v2.0 生产级韧性版）
# =============================================================================
# 文本通道（DEEPSEEK_*）与视觉通道（VISION_*）完全解耦。
#
# 生产韧性设计（对应真实用户场景）：
# 1. 错误语义分类——不是所有 429 都该重试：智谱 code=1113（余额不足）重试
#    纯浪费（实测白等 27s），必须立即失败并给出用户行动指引；限流/5xx/断网
#    才是可重试。
# 2. 有界重试 + 指数退避：最多 max_retries 次，退避 base*2^n。
# 3. 熔断器：连续 threshold 次最终失败后开闸 cooldown 秒——上游挂掉时后续
#    请求毫秒级快速失败（不再逐个吃满重试雪上加霜），冷却后半开自动恢复。
# 4. 健康可观测：status() 供 /health 与运维查看（配置态/熔断态/最近错误）。
# =============================================================================

import base64
import threading
import time
from pathlib import Path

import requests

from app.config import (
    VISION_API_KEY, VISION_BASE_URL, VISION_DESCRIBE_PROMPT,
    VISION_MODEL, VISION_TIMEOUT_SECONDS,
)

MAX_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# 智谱错误码 → 不可重试 + 用户行动指引（资源/账户类故障重试无意义）
NON_RETRYABLE_HINTS = {
    "1113": "余额不足或无可用资源包——请到智谱控制台充值或领取免费资源包后重试",
    "1211": "API key 无效或已过期——请检查 VISION_API_KEY",
    "1301": "模型未开通或无权限——请确认该账号已开通此视觉模型",
}


class VisionUnavailableError(RuntimeError):
    """视觉专线未配置（VISION_API_KEY 为空）。"""


class VisionError(RuntimeError):
    """视觉专线调用失败。携带错误类别与用户行动指引。"""

    def __init__(self, message: str, retryable: bool = False, provider_code: str = ""):
        super().__init__(message)
        self.retryable = retryable
        self.provider_code = provider_code


class VisionService:
    def __init__(self, api_key: str = VISION_API_KEY, base_url: str = VISION_BASE_URL,
                 model: str = VISION_MODEL, timeout: float = VISION_TIMEOUT_SECONDS,
                 max_retries: int = 3, retry_base_delay: float = 1.0,
                 circuit_threshold: int = 3, circuit_cooldown: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.circuit_threshold = circuit_threshold
        self.circuit_cooldown = circuit_cooldown
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._last_error: str | None = None

    # ── 状态与健康 ──

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict:
        with self._lock:
            circuit = "open" if self._circuit_open_until > time.time() else "closed"
            return {
                "configured": self.configured,
                "model": self.model,
                "circuit": circuit,
                "consecutive_failures": self._consecutive_failures,
                "last_error": self._last_error,
            }

    def _record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._last_error = None

    def _record_failure(self, message: str):
        with self._lock:
            self._consecutive_failures += 1
            self._last_error = message
            if self._consecutive_failures >= self.circuit_threshold:
                self._circuit_open_until = time.time() + self.circuit_cooldown

    # ── 核心调用 ──

    def describe_image(self, image_path: str, question: str = VISION_DESCRIBE_PROMPT) -> str:
        if not self.configured:
            raise VisionUnavailableError(
                "视觉专线未配置：在 .env 里填 VISION_API_KEY/VISION_BASE_URL/VISION_MODEL"
                "（如智谱 glm-4v-flash）后重试"
            )

        with self._lock:
            if self._circuit_open_until > time.time():
                remain = int(self._circuit_open_until - time.time())
                raise VisionError(
                    f"视觉上游熔断中（连续 {self._consecutive_failures} 次失败），"
                    f"{remain}s 后自动半开重试。最近错误: {self._last_error}"
                )

        path = Path(image_path)
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            raise VisionError(f"不支持的图片格式: {path.suffix}，仅支持 {SUPPORTED_IMAGE_EXTS}")
        raw = path.read_bytes()
        if len(raw) > MAX_IMAGE_BYTES:
            raise VisionError(f"图片超过 {MAX_IMAGE_BYTES // 1024 // 1024}MB 上限")

        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

        last_error = "未知错误"
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": question},
                            ],
                        }],
                        "max_tokens": 1024,
                    },
                    timeout=self.timeout,
                )
            except requests.Timeout:
                last_error = f"视觉端点超时（{self.timeout}s，第 {attempt + 1} 次）"
                # 断网/上游挂起：可重试路径
            except requests.ConnectionError as e:
                last_error = f"断网或无法连接视觉端点: {str(e)[:120]}"
                # 可重试路径
            else:
                if resp.status_code == 200:
                    self._record_success()
                    try:
                        return resp.json()["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        self._record_failure(f"响应结构异常: {e}")
                        raise VisionError(f"视觉端点响应结构异常: {e}") from e

                # 错误语义分类：看响应体里的 provider code（比裸状态码准）
                provider_code, hint = self._classify(resp)
                if hint and not hint.startswith("限流"):
                    # 资源/账户类故障：重试无意义，立即失败 + 行动指引
                    self._record_failure(f"HTTP {resp.status_code} [{provider_code}] {hint}")
                    raise VisionError(
                        f"视觉专线不可用（{provider_code}）: {hint}", provider_code=provider_code
                    )
                last_error = f"HTTP {resp.status_code} [{provider_code}] {hint or resp.text[:120]}"
                # 429 限流 / 5xx：可重试

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_base_delay * (2 ** attempt))

        self._record_failure(last_error)
        raise VisionError(
            f"视觉专线连续 {self.max_retries} 次失败: {last_error}", retryable=True
        )

    @staticmethod
    def _classify(resp) -> tuple[str, str]:
        """从响应体提取 provider code 与（可判断时）用户行动指引。"""
        code, hint = str(resp.status_code), ""
        try:
            data = resp.json()
            code = str(data.get("error", {}).get("code", code))
            for key, action in NON_RETRYABLE_HINTS.items():
                if code == key:
                    hint = action
                    break
            else:
                if resp.status_code == 429:
                    hint = "限流"
                elif resp.status_code >= 500:
                    hint = "上游服务故障"
        except Exception:
            pass
        return code, hint
