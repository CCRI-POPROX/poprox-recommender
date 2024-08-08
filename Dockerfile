ARG LOG_LEVEL=INFO
# Use Lambda "Provided" base image for the build container
FROM public.ecr.aws/lambda/provided:al2 as build

# install necessary system packages
RUN yum -y install git-core

# Fetch the micromamba executable from GitHub
ADD --chmod=0755 https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# Copy dependency lockfile and install deps
COPY conda-lock.yml /src/poprox-recommender/
RUN micromamba create -y --always-copy -p /opt/poprox -f /src/poprox-recommender/conda-lock.yml
# Download the punkt NLTK data
RUN micromamba run -p /opt/poprox python -m nltk.downloader -d /opt/poprox/nltk_data punkt
# Install the Lambda runtime bridge
RUN micromamba install -p /opt/poprox -c conda-forge awslambdaric

# Copy the soure code into the image to install it
# TODO do we want to copy the sdist or wheel instead?
COPY pyproject.toml README.md LICENSE.md /src/poprox-recommender/
COPY src/ /src/poprox-recommender/src/
WORKDIR /src/poprox-recommender
# Install the poprox-recommender module
RUN micromamba run -p /opt/poprox pip install --no-deps .

# Use Lambda "Provided" base image for the deployment container
# We installed Python ourselves
FROM public.ecr.aws/lambda/provided:al2

# Copy the installed packages and data from the build container
COPY --from=build /usr/local/bin/micromamba /usr/local/bin/micromamba
COPY --from=build /opt/poprox /opt/poprox
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# Bake the model data into the image
COPY models/ /opt/poprox/models/
ENV POPROX_MODELS=/opt/poprox/models

# Make sure we can import the recommender
RUN micromamba run -p /opt/poprox python -m poprox_recommender.handler

# Copy the bootstrap script
COPY --chmod=0555 lambda-bootstrap.sh /var/runtime/bootstrap

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV AWS_LAMBDA_LOG_LEVEL=${LOG_LEVEL}

# Run the bootstrap with our handler by default
CMD ["poprox_recommender.handler.generate_recs"]
