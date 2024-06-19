#!/bin/sh

# TODO: add option to update only our dependencies once conda-lock bug is fixed
# bug url: https://github.com/conda/conda-lock/issues/652

# make sure we're in the right directory
if [ ! -f pyproject.toml ]; then
    echo "this script must be run from the project root" >&2
    exit 64
fi

# our dependencies require flexible resolution to work on Windows, because
# conda-forge does not ship Windows binaries for current Torch versions.
# If users have followed the Conda recommended defaults to enable strict
# channel priority, locking will fail, so we override that setting here.
export CONDA_CHANNEL_PRIORITY=flexible

# and run conda-lock
exec conda-lock lock -f pyproject.toml -f dev/environment.yml -f dev/constraints.yml
