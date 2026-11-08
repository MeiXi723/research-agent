import os
from unittest.mock import patch


def test_config_reads_env_vars():
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["TAVILY_API_KEY"] = "test-tavily"
    os.environ["MAX_AGENT_STEPS"] = "5"
    try:
        import importlib
        import app.config
        # config 用 load_dotenv(override=True)（.env 是唯一配置源）；本测试验证
        # 「config 读环境变量」这一层，mock 掉 dotenv 装载以隔离 .env 文件干扰
        with patch("dotenv.load_dotenv"):
            importlib.reload(app.config)

        assert app.config.DEEPSEEK_API_KEY == "test-key"
        assert app.config.TAVILY_API_KEY == "test-tavily"
        assert app.config.MAX_AGENT_STEPS == 5
    finally:
        # 还原真实 .env 口径，避免 test-key 残留在 os.environ 污染同进程后续测试
        import importlib
        import app.config
        importlib.reload(app.config)


def test_config_defaults():
    for key in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K"]:
        os.environ.pop(key, None)

    import importlib
    import app.config
    importlib.reload(app.config)

    assert app.config.CHUNK_SIZE == 500
    assert app.config.CHUNK_OVERLAP == 50
    assert app.config.TOP_K == 3
