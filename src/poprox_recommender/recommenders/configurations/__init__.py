"""
Module definitions for the different recommender pipelines defined in the POPROX
recommender service.
"""

DEFAULT_PIPELINE = "nrms_topics_fb_filter"
"""
The default recommender pipeline. Can be overridden by the
:envvar:`POPROX_DEFAULT_PIPELINE` environment variable.
"""
