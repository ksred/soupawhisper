.PHONY: run run-stream run-no-stream test help

run:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-stream:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --streaming; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --streaming; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-stream-verbose:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --streaming --verbose; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --streaming --verbose; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

run-no-stream:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python dictate.py --no-streaming; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python dictate.py --no-streaming; \
	else \
		echo "Error: Neither uv nor poetry found. Please install one of them."; \
		exit 1; \
	fi

test:
	@if command -v uv >/dev/null 2>&1; then \
		if [ ! -d .venv ]; then \
			uv venv .venv; \
		fi; \
		uv pip install --python .venv/bin/python -e . && \
		uv pip install --python .venv/bin/python "pytest>=8.0.0" "pytest-timeout>=2.0.0" && \
		.venv/bin/pytest dictate_tests.py; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry install && \
		poetry run pytest dictate_tests.py; \
	else \
		if [ ! -d .venv ]; then \
			python3 -m venv .venv; \
		fi; \
		.venv/bin/pip install -e . pytest && \
		.venv/bin/pytest dictate_tests.py; \
	fi
