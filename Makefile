.PHONY: docs docs-install

docs-install:
	uv pip install -e ".[docs]"

docs: docs-install
	uv run --extra docs mkdocs serve -f docs/mkdocs.yml
