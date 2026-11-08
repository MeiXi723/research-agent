.PHONY: install test e2e lint cov run recall smoke-live docker-build clean

# Windows Git Bash venv 布局；Linux/macOS 改 .venv/bin/python
PY := .venv/Scripts/python.exe
PIP_MIRROR := https://mirrors.aliyun.com/pypi/simple/

install:
	python -m venv .venv
	$(PY) -m pip install -i $(PIP_MIRROR) -r requirements.txt

test:
	$(PY) -m pytest

e2e:
	$(PY) -m pytest -m e2e

lint:
	$(PY) -m ruff check app mcp_servers evals tests

cov:
	$(PY) -m pytest --cov=app --cov-report=term-missing

run:
	$(PY) -m uvicorn app.main:app --port 8000 --reload

# 评测：recall 离线可复算（首跑需 fastembed 模型缓存）
recall:
	$(PY) evals/run_eval.py --mode recall

# 真实 LLM 对照评测（需 .env 里 DEEPSEEK_API_KEY）
compare:
	$(PY) evals/run_eval.py --mode compare --n 10

# 评测数据集拉取（opt-in，需 pip install datasets）
pull-evals:
	$(PY) -m pip install -i $(PIP_MIRROR) datasets
	$(PY) evals/pull_datasets.py --n 50

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} \; 2>/dev/null || true
