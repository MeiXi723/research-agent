# =============================================================================
# vision.py — 多模态视觉专线（v2.0）
# =============================================================================
# 文本通道（DEEPSEEK_*）与视觉通道（VISION_*）完全解耦：
# - 文本模型没视觉能力，图片理解走独立提供商（GLM-4V / doubao-vision / qwen-vl，
#   任意 OpenAI 兼容视觉端点）
# - 统一 OpenAI 兼容格式：messages[].content 数组 + image_url(base64 data URL)
# - VISION_API_KEY 为空 = 专线未配置，调用方拿到 VisionUnavailableError 优雅降级
# =============================================================================

import base64
from pathlib import Path

import requests

from app.config import (
    VISION_API_KEY, VISION_BASE_URL, VISION_DESCRIBE_PROMPT,
    VISION_MODEL, VISION_TIMEOUT_SECONDS,
)

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # base64 前的原图上限
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


class VisionUnavailableError(RuntimeError):
    """视觉专线未配置（VISION_API_KEY 为空）。调用方据此优雅降级。"""


class VisionError(RuntimeError):
    """视觉专线已配置但调用失败（网络/模型/配额）。携带可操作信息。"""


class VisionService:
    def __init__(self, api_key: str = VISION_API_KEY, base_url: str = VISION_BASE_URL,
                 model: str = VISION_MODEL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def describe_image(self, image_path: str, question: str = VISION_DESCRIBE_PROMPT) -> str:
        """图片 → 文本描述（可直接入库/检索）。真实 HTTP 调用视觉端点。"""
        if not self.configured:
            raise VisionUnavailableError(
                "视觉专线未配置：在 .env 里填 VISION_API_KEY/VISION_BASE_URL/VISION_MODEL"
                "（如智谱 glm-4v-flash）后重试"
            )

        path = Path(image_path)
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            raise VisionError(f"不支持的图片格式: {path.suffix}，仅支持 {SUPPORTED_IMAGE_EXTS}")
        raw = path.read_bytes()
        if len(raw) > MAX_IMAGE_BYTES:
            raise VisionError(f"图片超过 {MAX_IMAGE_BYTES // 1024 // 1024}MB 上限")

        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

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
            timeout=VISION_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            # 把提供商的错误码透传出来——配额/鉴权/限流各有不同处理方式
            raise VisionError(f"视觉端点 HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise VisionError(f"视觉端点响应结构异常: {e}") from e
