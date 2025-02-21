#!/bin/bash
set -xeo pipefail

# set up environment dirs
sudo chown vscode:vscode .pixi node_modules

# fix git permissions warning
git config --global --add safe.directory $PWD

# get pre-commit wired up and ready
pre-commit install
pre-commit install-hooks

# install the development environment
pixi install -e dev
