#!/bin/bash

set -eo pipefail

repo=""
upload=yes

while [[ -n $1 ]]; do
    case "$1" in
    --shared)
        repo=shared
        shift
        ;;
    --public)
        repo=public
        shift
        ;;
    -n | --dry-run)
        upload=no
        shift
        ;;
    --*)
        echo "unknown option $1" >&2
        exit 2
        ;;
    *)
        echo "unrecognized command-line option $1" >&2
        exit 2
        ;;
    esac
done

if [[ -z $repo ]]; then
    echo "no target specified" >&2
    exit 2
fi

# define our commands for sharing
public() {
    # upload public files to all repos
    echo "uploading public files"
    echo + dvc push -r $repo -R "$@"
    if [[ $upload == yes ]]; then
        dvc push -r $repo "$@"
    fi
}

shared() {
    if [[ $repo == shared ]]; then
        echo "uploading shared files"
        echo + dvc push -r $repo -R "$@"
        if [[ $upload == yes ]]; then
            dvc push -r $repo "$@"
        fi
    fi
}

. shared-data.sh
