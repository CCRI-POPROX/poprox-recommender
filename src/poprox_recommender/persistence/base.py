from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from poprox_concepts.domain import RecommendationList


class PersistenceManager(ABC):
    """
    Abstract base class for persisting pipeline data.
    
    Provides interface for saving and loading user models, recommendation lists,
    and associated metadata from the recommendation pipeline.
    """

    @abstractmethod
    def save_pipeline_data(
        self,
        request_id: str,
        user_model: str,
        original_recommendations: RecommendationList,
        rewritten_recommendations: RecommendationList,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save all pipeline data for a recommendation session.
        
        Args:
            request_id: Unique identifier for this request
            user_model: Generated user model text
            original_recommendations: Ranked recommendations before rewriting
            rewritten_recommendations: Final recommendations with rewritten headlines
            metadata: Additional metadata to store
            
        Returns:
            session_id: Unique identifier for this saved session
        """
        pass

    @abstractmethod
    def load_pipeline_data(self, session_id: str) -> Dict[str, Any]:
        """
        Load previously saved pipeline data.

        Args:
            session_id: Session identifier returned from save_pipeline_data
            
        Returns:
            Dictionary containing all saved pipeline data
        """
        pass

    @abstractmethod
    def load_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Load only the metadata for a persisted session.

        Args:
            session_id: Session identifier returned from save_pipeline_data

        Returns:
            Dictionary containing the stored metadata
        """
        pass

    @abstractmethod
    def list_sessions(self, request_id_prefix: Optional[str] = None) -> list[str]:
        """
        List available saved sessions.

        Args:
            request_id_prefix: Optional prefix to filter sessions
            
        Returns:
            List of session IDs
        """
        pass
