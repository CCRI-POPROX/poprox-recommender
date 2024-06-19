#!/bin/sh

export CONDA_CHANNEL_PRIORITY=flexible
if [ ! -f pyproject.toml ]; then
    echo "this script must be run from the project root" >&2
    exit 64
fi

exec conda-lock lock -f pyproject.toml -f dev/environment.yml -f dev/constraints.yml
