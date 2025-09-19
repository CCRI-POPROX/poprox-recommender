#!/bin/bash
set -xeo pipefail

# fix git permissions warning
git config --global --add safe.directory $PWD

# install the development environment
uv venv
uv sync --group cpu

# get pre-commit wired up and ready
uv run pre-commit install
uv run pre-commit install-hooks
