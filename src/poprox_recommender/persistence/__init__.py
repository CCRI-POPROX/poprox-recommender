import os

from .base import PersistenceManager
from .local import LocalPersistenceManager

# Default S3 bucket for persistence
DEFAULT_PERSISTENCE_BUCKET = "poprox-default-recommender-pipeline-data-prod"


def get_persistence_manager() -> PersistenceManager:
    """
    Factory function to get the appropriate persistence manager based on environment.

    Returns:
        PersistenceManager: LocalPersistenceManager for local dev, S3PersistenceManager for Lambda
    """
    backend_override = os.getenv("PERSISTENCE_BACKEND")
    backend = backend_override.lower() if backend_override else None

    if backend not in (None, "auto", "s3", "local"):
        raise ValueError(f"Unsupported PERSISTENCE_BACKEND value: {backend_override}")

    use_s3 = False
    if backend == "s3":
        use_s3 = True
    elif backend == "local":
        use_s3 = False
    else:
        # Auto mode defaults to S3 when running in Lambda
        use_s3 = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

    if use_s3:
        from .s3 import S3PersistenceManager

        bucket = os.getenv("PERSISTENCE_BUCKET", DEFAULT_PERSISTENCE_BUCKET)
        prefix = os.getenv("PERSISTENCE_PREFIX", "pipeline-outputs/")
        return S3PersistenceManager(bucket, prefix)

    persistence_path = os.getenv("PERSISTENCE_PATH", "./data/pipeline_outputs")
    return LocalPersistenceManager(persistence_path)


__all__ = [
    "PersistenceManager",
    "LocalPersistenceManager",
    "get_persistence_manager",
    "DEFAULT_PERSISTENCE_BUCKET",
]
