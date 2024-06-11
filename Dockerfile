# Use Lambda "Provided" base image for the build container
FROM public.ecr.aws/lambda/provided:al2 as build

# install necessary system packages
RUN yum -y install git

# Fetch the micromamba executable from GitHub
ADD --chmod=0755 https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 /usr/local/bin/micromamba

# Copy source and dependency specifications
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# copy the lock file to install deps
COPY conda-lock.yml /tmp/conda-lock.yml
# Install recommender dependencies from Conda
RUN micromamba create -y --always-copy -p /opt/poprox -f /tmp/conda-lock.yml
# Download the punkt NLTK data
RUN micromamba run -p /opt/poprox python -m nltk.downloader -d /opt/poprox/nltk_data punkt
# Install the Lambda runtime bridge
RUN micromamba install -p /opt/poprox -c conda-forge awslambdaric

# Copy the source and models to bake into the model
COPY ./ /src/poprox_recommender
WORKDIR /src/poprox_recommender
# Install the poprox-recommender module
RUN micromamba run -p /opt/poprox pip install .

# Use Lambda "Provided" base image for the deployment container
# We installed Python ourselves
FROM public.ecr.aws/lambda/provided:al2

# Copy the installed packages and data from the build container
COPY --from=build /usr/local/bin/micromamba /usr/local/bin/micromamba
COPY --from=build /opt/poprox /opt/poprox
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE /tmp/.transformers

# Since we use the "provided" runtime, we need to set Docker entry point
ENTRYPOINT "/bin/bash"
# ENTRYPOINT ["/usr/local/bin/micromamba", "run", "-p", "/opt/poprox", "python", "-m", "awslambdaric"]
# Tell it to use our poprox recommender entry point
# CMD ["poprox_recommender.handler.generate_recs"]
