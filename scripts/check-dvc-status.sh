#!/bin/bash
set -eo pipefail

report_file=dvc-status.log
status_file=$(mktemp --tmpdir poprox-dvc-status.XXXXXXXX)
trap 'rm $status_file' INT TERM EXIT

dvc status --no-updates --json >$status_file

n_changed=$(jq length <$status_file)
echo "changed=$n_changed" >>"$GITHUB_OUTPUT"

if [[ $n_changed -eq 0 ]]; then
    echo "::notice::DVC pipeline is up-to-date"
    cat >$report_file <<EOF
✅ The DVC pipeline is up-to-date.

Creator: check-dvc-status
EOF
    exit 0
fi

# Prepare a report for the out-of-date information.
echo "::notice::$n_changed stages are out-of-date"
cat >$report_file <<EOF
🚨 The DVC pipeline is out-of-date. 🚨

This is not a hard error, but the DVC-controlled outputs in this PR, such as
evaluation metrics, are not current with respect to their code and data inputs.

If the MIND eval CI job also fails, then the pipeline is not only out-of-date but
cannot be rerun to produce current outputs.

\`\`\`console
$ dvc status
EOF

dvc status --no-updates | tee -a $report_file
echo -e '```\n' >>$report_file
echo -e 'Creator: check-dvc-status' >>$report_file

# emit GitHub error messages attached to each individual stage
jq -r 'keys | .[] | sub(":"; " ")' <$status_file | (while read file stage; do
    if [[ -z $stage ]]; then
        stage="$file"
        file="dvc.yaml"
    fi
    grep -n "$stage:" "$file" | sed -Ee "s#([[:digit:]]+):.*#::error file=$file,line=\1,title=Stage out-of-date::The DVC stage $file/$stage is out-of-date#"
done)

echo "::error::$n_changed stages are out-of-date"
