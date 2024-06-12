#!/bin/sh

args="$@"
# this automatically launches either the emulator or the runtime
# see: https://github.com/aws/aws-lambda-runtime-interface-emulator
# first, set up our Python environment
eval "$(micromamba shell activate -s dash -p /opt/poprox)"
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
    # not on AWS, use the emulator
    exec /usr/local/bin/aws-lambda-rie python -m awslambdaric $args
else
    exec python -m awslambdaric $args
fi
