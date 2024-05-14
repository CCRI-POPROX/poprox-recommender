# Use Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11

# Install prerequisite packages
RUN yum -y install git gcc python3-devel

# Install Python dependencies w/ special requirements
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install nltk && python -m nltk.downloader -d /var/lang/nltk_data punkt

# Install the package and other dependencies
COPY ./ ${LAMBDA_TASK_ROOT}/poprox_recommender
RUN pip install ${LAMBDA_TASK_ROOT}/poprox_recommender

# Set entry point
CMD ["poprox_recommender.handler.generate_recs"]
