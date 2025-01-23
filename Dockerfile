# Use Lambda "Provided" base image for the build container
FROM public.ecr.aws/lambda/provided:al2023 AS build
ARG PIXI_VERSION=0.40.3

# install necessary system packages
RUN dnf -y install git-core

# Fetch the pixi executable from GitHub
# see: https://github.com/prefix-dev/pixi-docker/blob/main/Dockerfile
RUN curl -fsL \
    "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl" \
    -o /usr/local/bin/pixi && chmod +x /usr/local/bin/pixi

# Copy the soure code into the image to install it and create the environment
# TODO do we want to copy the sdist or wheel instead?
COPY pyproject.toml pixi.toml pixi.lock README.md LICENSE.md /src/poprox-recommender/
COPY src/ /src/poprox-recommender/src/
WORKDIR /src/poprox-recommender
RUN mkdir build

# install the production environment (will include poprox-recommender)
ENV PIXI_LOCKED=true
RUN pixi install -e production
RUN pixi install -e pkg
# Download the punkt NLTK data
RUN pixi run -e production python -m nltk.downloader -d build/nltk_data punkt
# Install poprox-recommender
RUN pixi run -e production pip install --no-deps --root-user-action=ignore .
# Pack up the environment for migration to runtime
RUN ./.pixi/envs/pkg/bin/conda-pack -p .pixi/envs/production -d /opt/poprox -o build/production-env.tar

# Use Lambda "Provided" base image for the deployment container
# We installed Python ourselves
FROM public.ecr.aws/lambda/provided:al2023
ARG LOG_LEVEL=INFO

# Unpack the packaged environment from build container into runtime contianer
# GNU tar chokes on conda-pack's output, so we use bsdtar
RUN dnf install -y bsdtar && dnf clean -y all
RUN mkdir /opt/poprox
RUN --mount=type=bind,from=build,source=/src/poprox-recommender/build,target=/tmp/poprox-build \
    bsdtar -C /opt/poprox -xf /tmp/poprox-build/production-env.tar

# Copy the fetched NLTK data into the runtime container
COPY --from=build /src/poprox-recommender/build/nltk_data /opt/poprox/nltk_data

# Bake the model data into the image
COPY models/ /opt/poprox/models/
ENV POPROX_MODELS=/opt/poprox/models

# Make sure we can import the recommender
RUN /opt/poprox/bin/python -m poprox_recommender.handler

# Copy the bootstrap script
COPY --chmod=0555 lambda-bootstrap.sh /var/runtime/bootstrap

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV AWS_LAMBDA_LOG_LEVEL=${LOG_LEVEL}

# Run the bootstrap with our handler by default
CMD ["poprox_recommender.handler.generate_recs"]
