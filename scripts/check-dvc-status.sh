#!/bin/bash
set -eo pipefail

status_file=$(mktemp --tmpdir poprox-dvc-status.XXXXXXXX)
trap 'rm $status_file' INT TERM EXIT

dvc status --no-updates --json >$status_file

n_changed=$(jq length <$status_file)
echo "$n_changed stages have changed"
if [[ $n_changed -eq 0 ]]; then
    echo "::notice::DVC pipeline is up-to-date"
    exit 0
fi

# emit GitHub error messages attached to each individual stage
jq -r 'keys | .[] | sub(":"; " ")' <$status_file | (while read file stage; do
    if [[ -z $stage ]]; then
        stage="$file"
        file="dvc.yaml"
    fi
    grep -n "$stage:" "$file" | sed -Ee "s#([[:digit:]]+):.*#::error file=$file,line=\1,title=Stage out-of-date::The DVC stage $file/$stage is out-of-date#"
done)

echo "::error::$n_changed stages are out-of-date"
exit 1
