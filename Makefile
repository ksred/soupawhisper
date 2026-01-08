.PHONY: run run-stream run-no-stream test help service-stop service-start service-restart service-status service-logs record

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

service-stop:
	@systemctl --user stop soupawhisper || echo "Service not running or not installed"

service-start:
	@systemctl --user start soupawhisper || echo "Service not installed"

service-restart:
	@systemctl --user restart soupawhisper || echo "Service not installed"

service-status:
	@systemctl --user status soupawhisper || echo "Service not installed"

service-logs:
	@journalctl --user -u soupawhisper -f || echo "Service not installed"

record:
	@if command -v arecord >/dev/null 2>&1; then \
		FILENAME="/tmp/recording_$$(date +%Y%m%d_%H%M%S).wav"; \
		echo "Recording to $$FILENAME (Press Ctrl+C to stop)..."; \
		arecord -f S16_LE -r 16000 -c 1 -t wav "$$FILENAME"; \
		echo "Recording saved to $$FILENAME"; \
	else \
		echo "Error: arecord not found. Please install alsa-utils."; \
		exit 1; \
	fi
