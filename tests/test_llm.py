import numpy as np
from unittest.mock import MagicMock, patch
from app.services.llm import LLMService


@patch("app.services.llm.TextEmbedding")
@patch("app.services.llm.OpenAI")
def test_embed(mock_openai, mock_embedding):
    mock_embed_model = MagicMock()
    mock_embed_model.embed.return_value = [np.array([0.1, 0.2, 0.3])]
    mock_embedding.return_value = mock_embed_model

    svc = LLMService(api_key="sk-test", base_url="https://test.com")
    result = svc.embed("测试文本")

    assert isinstance(result, list)
    assert len(result) == 3


@patch("app.services.llm.TextEmbedding")
@patch("app.services.llm.OpenAI")
def test_chat(mock_openai, mock_embedding):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "你好，我是 AI 助手"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    mock_openai.return_value = mock_client

    mock_embedding.return_value = MagicMock()

    svc = LLMService(api_key="sk-test", base_url="https://test.com")
    response = svc.chat([{"role": "user", "content": "你好"}])

    assert response == "你好，我是 AI 助手"
