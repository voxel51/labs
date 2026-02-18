# CI Testing Infrastructure

This directory contains GitHub Actions workflows for testing FiftyOne Labs plugins.

## Overview

The CI infrastructure ensures that all plugins in the repository:
- Are compatible with the latest (or pinned) FiftyOne version
- Work correctly with cloud media (future)
- Only use dependencies available in the `fiftyone-teams-cv-full` Docker image (future)

## Workflows

### `plugin-tests.yml`

Main workflow for running plugin tests.

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

**Features:**
- Matrix strategy to test each plugin in parallel
- Automatic test discovery in `plugins/*/tests/` directories
- Coverage reporting via Codecov

**Runner:**
- Uses GitHub-hosted `ubuntu-latest` runners
- Free for public repositories (unlimited minutes)
- 2,000 free minutes/month for private repositories

## Test Structure

Each plugin should have a `tests/` directory with the following structure:

```
plugins/<plugin-name>/
├── __init__.py
├── fiftyone.yml
└── tests/
    ├── __init__.py
    ├── conftest.py          # Shared fixtures
    ├── test_compatibility.py  # FO version compatibility tests
    ├── test_cloud_media.py     # Cloud media tests (future)
    └── test_dependencies.py    # Dependency validation (future)
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring FO instance
- `@pytest.mark.cloud` - Tests requiring cloud credentials
- `@pytest.mark.slow` - Tests that take longer to run

## Required Secrets (Future)

When cloud media testing is implemented, the following secrets will be required:

### Cloud Storage Credentials

- `AWS_ACCESS_KEY_ID` - For S3-backed dataset testing
- `AWS_SECRET_ACCESS_KEY` - For S3-backed dataset testing
- `GCS_CREDENTIALS_JSON` - For GCS-backed dataset testing (if needed)
- `FIFTYONE_TEAMS_API_KEY` - For FiftyOne Teams cloud storage testing

### Optional Secrets

- `FIFTYONE_TEAMS_URI` - Teams instance URI (if testing against specific instance)
- `TEST_S3_BUCKET` - S3 bucket name for test datasets
- `TEST_GCS_BUCKET` - GCS bucket name for test datasets

## Adding a Plugin to CI

To add a new plugin to the CI test matrix:

1. Create a `tests/` directory in your plugin folder
2. Add at minimum `test_compatibility.py` with FO version compatibility tests
3. Add your plugin name to the matrix in `.github/workflows/plugin-tests.yml`:

```yaml
matrix:
  plugin:
    - template
    - your_plugin_name  # Add here
```

## Running Tests Locally

To run tests locally, use pytest:

```bash
# Run all tests for a specific plugin
pytest plugins/template/tests/

# Run only compatibility tests
pytest plugins/template/tests/test_compatibility.py

# Run with coverage
pytest plugins/template/tests/ --cov=plugins/template --cov-report=html
```

## Test Requirements

### Initial Implementation (Current)

- ✅ FO version compatibility tests (`test_compatibility.py`)
- ⏳ Cloud media tests (`test_cloud_media.py`) - table for later
- ⏳ Dependency validation tests (`test_dependencies.py`) - table for later

### Future Enhancements

- Cloud media testing infrastructure
- Dependency validation against `fiftyone-teams-cv-full` Docker image
- Integration with Docker-based testing environment

## Reference

- Template plugin: `plugins/template/` - Example implementation with all test types
- Existing test example: `plugins/few_shot_learning/tests/test_models.py`
- Dockerfile reference: `https://github.com/voxel51/fiftyone-teams/blob/develop/deployment/docker/Dockerfile-teams-cv-full`
