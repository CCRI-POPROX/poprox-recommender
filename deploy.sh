#!/bin/bash

while getopts ":e:r:" flag; do
    case "${flag}" in
    e) env=${OPTARG} ;;
    r) region=${OPTARG} ;;
    *)
        echo "Invalid option. Only -e and -r are allowed" >&2
        exit 1
        ;;
    esac
done
env=${env:-prod}
region=${region:-us-east-1}

echo "ENV: $env"
echo "Region: $region"

dvc pull -R models || exit 2

# Build container and deploy functions
npx serverless deploy --stage "${env}" --region "${region}" || exit 3
