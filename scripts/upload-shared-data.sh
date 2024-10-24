#!/bin/bash

set -eo pipefail

repo="UNKNOWN"
file_attr_pattern=""
upload=yes

while [[ -n $1 ]]; do
    case "$1" in
    --shared)
        file_attr_pattern='poprox-sharing: (shared|public)$'
        repo=shared
        shift
        ;;
    --public)
        file_attr_pattern='poprox-sharing: public$'
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

if [[ -z $file_attr_pattern ]]; then
    echo "no target specified" >&2
    exit 2
fi

# find the files with the appropriate attribute
echo "listing shared files"
declare -a files
files=($(dvc list --dvc-only -R . |
    git check-attr --stdin poprox-sharing |
    grep -E "$file_attr_pattern" |
    sed -e 's/:.*$//'))

echo "found ${#files[@]} files to share"
if [[ $upload == yes ]]; then
    set -x
    dvc push -r $repo "${files[@]}"
fi
