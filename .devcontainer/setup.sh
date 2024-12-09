#!/bin/sh
set -xeo pipefail

pixi install -e dev
pixi run -e dev pre-commit install
pixi run -e dev pre-commit install-hooks
