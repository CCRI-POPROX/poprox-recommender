#!/bin/sh

# Installed as /var/runtime/bootstrap, which the Lambda image
# entrypoint expects to bootstrap and run the handler software.

# set up the Conda environment
. /opt/poprox/bin/activate
# run our code through the Lambda RIC handler
exec /opt/poprox/bin/python -m awslambdaric "$_HANDLER"
