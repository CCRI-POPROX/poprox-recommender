ARG LOG_LEVEL=INFO
# Use Lambda Python base image for the build container
FROM public.ecr.aws/lambda/python:3.11 as build

# Install prerequisite packages
RUN yum -y install git gcc python3-devel

# Install Python dependencies w/ special requirements
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install nltk && python -m nltk.downloader -d /var/lang/nltk_data punkt

# Install our package
COPY ./ ${LAMBDA_TASK_ROOT}/poprox_recommender
RUN pip install ${LAMBDA_TASK_ROOT}/poprox_recommender


# Use Lambda Python base image for the deployment container
FROM public.ecr.aws/lambda/python:3.11

# Copy the installed packages from the build container
COPY --from=build /var/lang/lib/python3.11/site-packages /var/lang/lib/python3.11/site-packages
COPY --from=build /var/lang/nltk_data /var/lang/nltk_data
COPY ./models/ /var/lang/poprox-models
ENV POPROX_MODELS /var/lang/poprox-models

# Set the transformers cache to a writeable directory
ENV TRANSFORMERS_CACHE=/tmp/.transformers
ENV AWS_LAMBDA_LOG_LEVEL=${LOG_LEVEL}
# Set entry point function
CMD ["poprox_recommender.handler.generate_recs"]
