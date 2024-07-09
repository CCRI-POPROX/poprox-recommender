#!/bin/bash

# TODO: add option to update only our dependencies once conda-lock bug is fixed
# bug url: https://github.com/conda/conda-lock/issues/652

freeze_concepts=no

# make sure we're in the right directory
if [[ ! -f pyproject.toml ]]; then
    echo "this script must be run from the project root" >&2
    exit 64
fi

while [[ -n "$1" ]]; do
    case "$1" in
    --freeze-concepts)
        freeze_concepts=yes
        shift
        ;;
    --*)
        echo "invalid flag $1" >&2
        exit 2
        ;;
    *)
        echo "unexpected argument $1" >&2
        exit 2
        ;;
    esac
done

declare -a lock_args
lock_args=(-f pyproject.toml -f dev/environment.yml)

# check if we want to freeze popropx-concepts
if [[ $freeze_concepts = yes ]]; then
    declare -a concept_urls
    # extract the currently-locked POPROX URL
    concept_urls=($(yq '.package | map(select(.name == "poprox-concepts")) | map(.url) | unique | join("\n")' conda-lock.yml))
    if [[ ${#concept_urls[@]} -eq 1 ]]; then
        echo "found locked poprox-concepts at $concept_urls"
    elif [[ ${#concept_urls[@]} -eq 0 ]]; then
        echo "ERROR: no poprox-concepts not found in lockfile" >&2
        exit 3
    else
        echo "WARNING: lockfile has multiple poprox-concepts versions" >&2
        for ver in "${concept_urls[@]}"; do
            echo "- $ver" >&2
        done
        echo "using: ${concept_urls[0]}" >&2
    fi

    cat >dev/poprox-concepts-dynamic-constraint.yml <<EOF
dependencies:
- pip:
    - ${concept_urls[0]}
EOF
    lock_args=("${lock_args[@]}" -f dev/poprox-concepts-dynamic-constraint.yml)
fi

# our dependencies require flexible resolution to work on Windows, because
# conda-forge does not ship Windows binaries for current Torch versions.
# If users have followed the Conda recommended defaults to enable strict
# channel priority, locking will fail, so we override that setting here.
export CONDA_CHANNEL_PRIORITY=flexible

# and run conda-lock
std_lock_args=("${lock_args[@]}" -f dev/constraints.yml)
echo "locking for dev and deploy environments" >&2
echo "+ conda-lock lock ${std_lock_args[*]}" >&2
conda-lock lock "${std_lock_args[@]}" || exit 10

echo "locking for CUDA environments"
cuda_lock_args=("${lock_args[@]}" -f dev/cuda-constraints.yml)
cuda_lock_args=("${cuda_lock_args[@]}" --lockfile conda-lock-cuda.yml)
echo "+ conda-lock lock ${cuda_lock_args[*]}" >&2
conda-lock lock "${cuda_lock_args[@]}" || exit 10
