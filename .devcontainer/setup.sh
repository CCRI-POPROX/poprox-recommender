#!/bin/bash
set -xeo pipefail

# set up environment dirs
sudo chown vscode:vscode .pixi node_modules

# get pre-commit wired up and ready
pre-commit install
pre-commit install-hooks

# install the development environment
pixi install -e dev
pixi run -e dev install-serverless
