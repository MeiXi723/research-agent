import pytest
from app.services.document import parse_file, chunk_text


def test_parse_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello World", encoding="utf-8")
    text = parse_file(str(f))
    assert text == "Hello World"


def test_parse_unsupported_extension(tmp_path):
    f = tmp_path / "test.jpg"
    f.write_text("")
    with pytest.raises(ValueError):
        parse_file(str(f))


def test_chunk_text_small():
    result = chunk_text("短文本", chunk_size=500)
    assert result == ["短文本"]


def test_chunk_text_large():
    text = "A" * 600 + "B" * 600
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert len(result) >= 2


def test_chunk_text_exact_boundary():
    text = "A" * 500
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert len(result) == 1
