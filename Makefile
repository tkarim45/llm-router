.PHONY: install test run
install:
	pip install -e ".[dev]"
test:
	pytest -q
run:
	llm-router --mock
