# Thin uv wrappers so agents and the author run identical commands.
# See AGENTS.md for the full workflow and per-chapter task specs in qa/.

PY_SOURCES = bookutils.py utils.py qa CH12_FLYNN_ML4DD/src CH12_FLYNN_ML4DD/scripts CH12_FLYNN_ML4DD/tests CH12_FLYNN_ML4DD/conftest.py

.PHONY: help env lint format callouts test test-ch12 execute-ch

help:
	@echo "Targets:"
	@echo "  make env             Create the 3.12 env and sync the dev + advanced extras"
	@echo "  make lint            Callout guard + ruff + black (check) on the Python surface"
	@echo "  make format          Auto-format the Python surface with black"
	@echo "  make callouts        Run only the callout-leakage guard on all notebooks"
	@echo "  make test            CH12 unit tests"
	@echo "  make execute-ch NN=01  Execute a chapter notebook via nbmake (e.g. NN=01)"

env:
	uv venv --python 3.12
	uv sync --extra advanced --extra dev

lint: callouts
	uv run ruff check $(PY_SOURCES)
	uv run black --check --line-length 100 $(PY_SOURCES)

format:
	uv run black --line-length 100 $(PY_SOURCES)

callouts:
	uv run python qa/check_callouts.py

test test-ch12:
	uv run pytest CH12_FLYNN_ML4DD -v

# Usage: make execute-ch NN=01
execute-ch:
	uv run pytest --nbmake --nbmake-timeout=1200 CH$(NN)_FLYNN_ML4DD.ipynb
