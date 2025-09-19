#!/bin/bash

dvc config --local cache.type hardlink

echo "fetching data for $GH_EVT on $GH_REPO"

if ! dvc pull -r public -R models tests; then
    if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
        echo '::warning::Public repository missing data, trying private'
        if ! dvc pull -R models tests; then
            echo '::error::Private repository not available or missing model/test data'
            exit 2
        fi
    else
        echo '::error::Public repository is missing model/test data'
        exit 3
    fi
fi

if ! dvc pull data/MINDsmall_dev.db; then
    echo '::warning::Private repository not available, some tests will be skipped'
fi
