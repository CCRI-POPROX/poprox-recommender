# Use Lambda "Provided" base image for the build container
FROM public.ecr.aws/lambda/provided:al2 as build

# install necessary system packages
RUN yum -y install git

# Fetch the micromamba executable from GitHub
ADD --chmod=0755 https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 /usr/local/bin/micromamba

# Copy source and dependency specifications
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# Copy dependency files to install deps.
COPY conda-lock.yml /src/poprox-recommender/
# Install recommender dependencies from Conda
RUN micromamba create -y --always-copy -p /opt/poprox -f /src/poprox-recommender/conda-lock.yml
# Download the punkt NLTK data
RUN micromamba run -p /opt/poprox python -m nltk.downloader -d /opt/poprox/nltk_data punkt
# Install the Lambda runtime bridge
RUN micromamba install -p /opt/poprox -c conda-forge awslambdaric

# Copy the source and models to bake into the model.
# This is separate to allow the dependency stages to be cached.
# TODO do we want to copy the sdist or wheel instead?
COPY pyproject.toml README.md /src/poprox-recommender
COPY src/ /src/poprox-recommender/src/
WORKDIR /src/poprox-recommender
# Install the poprox-recommender module
RUN micromamba run -p /opt/poprox pip install .

# Copy the Poprox models
COPY models/ /opt/poprox/models/

# Use Lambda "Provided" base image for the deployment container
# We installed Python ourselves
FROM public.ecr.aws/lambda/provided:al2

# Install the Amazon RIE emulator for easy debugging
ADD --chmod=0755 https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie \
    /usr/local/bin/aws-lambda-rie

# Copy the installed packages and data from the build container
COPY --from=build /usr/local/bin/micromamba /usr/local/bin/micromamba
COPY --from=build /opt/poprox /opt/poprox
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# Copy the entrypoint script
COPY entrypoint.sh /opt/poprox-entrypoint.sh

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE /tmp/.transformers

# Since we use the "provided" runtime, we need to set Docker entry point
ENTRYPOINT ["/opt/poprox-entrypoint.sh"]
# Tell it to use our poprox recommender entry point
CMD ["poprox_recommender.handler.generate_recs"]
