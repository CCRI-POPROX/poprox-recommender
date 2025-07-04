# Use Python Lambda base image for build
FROM public.ecr.aws/lambda/python:3.12 AS build

RUN dnf install -y git
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy the source code into the image to install it and create the environment
COPY pyproject.toml uv.lock README.md LICENSE.md .python-version /src/poprox-recommender/
COPY src/ /src/poprox-recommender/src/
WORKDIR /src/poprox-recommender

# Install directly to the Lambda task root instead of a separate venv
ENV UV_PROJECT_ENVIRONMENT=${LAMBDA_TASK_ROOT}
ENV UV_PYTHON=3.12
ENV UV_LOCKED=TRUE
RUN uv sync --no-editable --no-default-groups --extra cpu --extra deploy
# Install the local package properly
RUN uv pip install --system .

FROM public.ecr.aws/lambda/python:3.12
ARG LOG_LEVEL=INFO

# Accept the API key as a build-time argument and set it as a runtime env var
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ENV POPROX_MODELS=${LAMBDA_TASK_ROOT}/models

# Copy the installed packages from build stage
COPY --from=build ${LAMBDA_TASK_ROOT}/lib/python3.12/site-packages/ ${LAMBDA_TASK_ROOT}/


# Debug: Check what was actually installed
RUN ls -la ${LAMBDA_TASK_ROOT}/
RUN find ${LAMBDA_TASK_ROOT}/ -name "*poprox*" -type d || echo "No poprox directories found"
RUN python -c "import sys; print('Python path:'); [print(p) for p in sys.path]"

# Bake the model data into the image
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Bake the prompts into the image
COPY prompts/ ${LAMBDA_TASK_ROOT}/prompts/

# Try to import - if this fails, we'll see the debug info above
# RUN python -c "import poprox_recommender.api.main; print('Import successful')"

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV AWS_LAMBDA_LOG_LEVEL=${LOG_LEVEL}

# Set the handler
CMD ["poprox_recommender.api.main.handler"]
