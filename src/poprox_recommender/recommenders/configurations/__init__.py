"""
Module definitions for the different recommender pipelines defined in the POPROX
recommender service.
"""

DEFAULT_PIPELINE = "nrms_topic_scores"
"""
The default recommender pipeline. Can be overridden by the
:envvar:`POPROX_DEFAULT_PIPELINE` environment variable.
"""
