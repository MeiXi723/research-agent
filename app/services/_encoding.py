"""tiktoken 编码器惰性单例（独立模块便于测试 patch 与复用）。"""

_encoding = None
_encoding_failed = False


def get_encoding():
    global _encoding, _encoding_failed
    if _encoding_failed:
        return None
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True
            return None
    return _encoding
