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

# Check if a script name was provided as a command line argument
if [ -z "$1" ]; then
    echo "Error: Not sure whether to run cloudformation or serverless"
    exit 1
fi

if [ "$1" == "cf" ]; then
    aws cloudformation deploy --template-file cloudformation.yml --stack-name poprox-research-recommender-"${env}" --region "${region}" --capabilities CAPABILITY_NAMED_IAM
elif [ "$1" == "sls" ]; then
    # Download model artifacts
    dvc pull -R models

    # Build container and deploy functions
    npx serverless deploy --stage "${env}" --region "${region}"
fi
