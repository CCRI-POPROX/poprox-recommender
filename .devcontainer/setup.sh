#!/bin/bash
set -xeo pipefail

pre-commit install
pre-commit install-hooks
pixi install -e dev
