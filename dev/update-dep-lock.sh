#!/bin/sh

# TODO: add option to update only our dependencies once conda-lock bug is fixed
# bug url: https://github.com/conda/conda-lock/issues/652

export CONDA_CHANNEL_PRIORITY=flexible
if [ ! -f pyproject.toml ]; then
    echo "this script must be run from the project root" >&2
    exit 64
fi

exec conda-lock lock -f pyproject.toml -f dev/environment.yml -f dev/constraints.yml
