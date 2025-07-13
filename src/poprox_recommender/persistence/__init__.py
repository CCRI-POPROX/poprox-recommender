import os

from .base import PersistenceManager
from .local import LocalPersistenceManager


def get_persistence_manager() -> PersistenceManager:
    """
    Factory function to get the appropriate persistence manager based on environment.
    
    Returns:
        PersistenceManager: LocalPersistenceManager for local dev, S3PersistenceManager for Lambda
    """
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        # Running in Lambda - import S3 manager only when needed
        from .s3 import S3PersistenceManager
        bucket = os.getenv("PERSISTENCE_BUCKET", "poprox-pipeline-data")
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return S3PersistenceManager(bucket, prefix)
    else:
        # Local development
        persistence_path = os.getenv("PERSISTENCE_PATH", "./data/pipeline_outputs")
        return LocalPersistenceManager(persistence_path)


__all__ = ["PersistenceManager", "LocalPersistenceManager", "get_persistence_manager"]