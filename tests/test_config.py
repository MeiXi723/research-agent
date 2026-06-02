import os


def test_config_reads_env_vars():
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["TAVILY_API_KEY"] = "test-tavily"
    os.environ["MAX_AGENT_STEPS"] = "5"

    import importlib
    import app.config
    importlib.reload(app.config)

    assert app.config.DEEPSEEK_API_KEY == "test-key"
    assert app.config.TAVILY_API_KEY == "test-tavily"
    assert app.config.MAX_AGENT_STEPS == 5


def test_config_defaults():
    for key in ["CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K"]:
        os.environ.pop(key, None)

    import importlib
    import app.config
    importlib.reload(app.config)

    assert app.config.CHUNK_SIZE == 500
    assert app.config.CHUNK_OVERLAP == 50
    assert app.config.TOP_K == 3
