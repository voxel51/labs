# CI Testing Infrastructure

GitHub Actions workflows for testing FiftyOne Labs plugins.

## Workflow: `plugin-tests.yml`

**Triggers:** Push to `main`, PR to `main`, workflow_dispatch

**Matrix:** `(plugin, suite)` — regular and intensive for all/on-changes; dependencies for plugins with `requirements.txt`.

- **Regular** — `pytest tests/ --ignore=tests/intensive` (with coverage)
- **Intensive** — `pytest tests/intensive/` (path-dependent)
- **Dependencies** — `docker run python:3.11-slim pip install fiftyone + plugin requirements.txt`

**Runner:** `ubuntu-latest` (GitHub-hosted)

## Test Structure

```
plugins/<plugin-name>/
├── __init__.py
├── fiftyone.yml
├── requirements.txt    # optional; triggers dependencies suite
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_*.py              # Regular tests
    └── intensive/
        └── test_*.py          # Intensive tests (zoo datasets, etc.)
```

## Pytest Markers

- `unit` `integration` `intensive` `dependencies` `cloud`

## Adding a Plugin

1. Add to `ALL_PLUGINS` in the prepare job
2. Add path filter and INTENSIVE_PLUGINS condition (for intensive tests)
3. Plugins with `requirements.txt` get a dependencies suite automatically

## Running Locally

```bash
cd plugins/template
export PYTHONPATH="$(pwd)/..:$PYTHONPATH"

# Regular only
pytest tests/ --ignore=tests/intensive

# Intensive only
pytest tests/intensive/

# With coverage
pytest tests/ --ignore=tests/intensive --cov=. --cov-report=term

# Dependencies (Docker) — from repo root
docker run --rm \
  -v "$(pwd)/plugins/template:/plugin:ro" \
  -w /plugin \
  python:3.11-slim \
  bash -c "pip install -q fiftyone && pip install -r requirements.txt"
```

## ToDo

- Cloud media tests
