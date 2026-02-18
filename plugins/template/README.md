# Template Plugin

This is a template plugin used for testing the CI infrastructure for FiftyOne Labs plugins.

## Purpose

This plugin serves as a reference implementation and testing framework. The test suite in CI includes the following:
1. Basic operator functionality & unit tests (from the `tests/` directory)
2. Dependency installation and compatibility with the FOE docker image
3. Integration with cloud media (**ToDo**)
4. Intensive operator functionality (from the `tests/intensive/` directory)

## Adding tests to CI

For every new plugin, add the `plugin_name` to the following places:
1. `.github/workflows/plugin-tests.yml` --> `jobs` --> `steps` --> `filter`: add the path filter. This will ensure this plugin's tests are triggered when anything in this path changes.
2. In the same yml file, also add the plugin name to `ALL_PLUGINS` and to the env variable named `FILTER_<plugin_name>`