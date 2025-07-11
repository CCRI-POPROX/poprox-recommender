import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from poprox_concepts.domain import RecommendationList

from .base import PersistenceManager


class LocalPersistenceManager(PersistenceManager):
    """
    File-based persistence manager for local development and testing.
    
    Stores pipeline data as files in a local directory structure:
    - {session_id}/user_model.txt
    - {session_id}/original_recommendations.pkl
    - {session_id}/rewritten_recommendations.pkl
    - {session_id}/metadata.json
    """

    def __init__(self, base_path: str = "./data/pipeline_outputs"):
        """
        Initialize local persistence manager.
        
        Args:
            base_path: Base directory for storing pipeline outputs
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_pipeline_data(
        self,
        request_id: str,
        user_model: str,
        original_recommendations: RecommendationList,
        rewritten_recommendations: RecommendationList,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save pipeline data to local files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        session_id = f"{request_id}_{timestamp}"
        session_dir = self.base_path / session_id
        session_dir.mkdir(exist_ok=True)

        # Save user model as text
        (session_dir / "user_model.txt").write_text(user_model, encoding="utf-8")

        # Save recommendations as pickle files (preserves object structure)
        with open(session_dir / "original_recommendations.pkl", "wb") as f:
            pickle.dump(original_recommendations, f)

        with open(session_dir / "rewritten_recommendations.pkl", "wb") as f:
            pickle.dump(rewritten_recommendations, f)

        # Save metadata as JSON
        full_metadata = {
            "request_id": request_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_model_length": len(user_model),
            "num_articles": len(original_recommendations.articles),
            "pipeline_type": "llm_rank_rewrite",
            **(metadata or {}),
        }

        with open(session_dir / "metadata.json", "w") as f:
            json.dump(full_metadata, f, indent=2)

        return session_id

    def load_pipeline_data(self, session_id: str) -> Dict[str, Any]:
        """Load pipeline data from local files."""
        session_dir = self.base_path / session_id

        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        # Load user model
        user_model = (session_dir / "user_model.txt").read_text(encoding="utf-8")

        # Load recommendations
        with open(session_dir / "original_recommendations.pkl", "rb") as f:
            original_recommendations = pickle.load(f)

        with open(session_dir / "rewritten_recommendations.pkl", "rb") as f:
            rewritten_recommendations = pickle.load(f)

        # Load metadata
        with open(session_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        return {
            "user_model": user_model,
            "original_recommendations": original_recommendations,
            "rewritten_recommendations": rewritten_recommendations,
            "metadata": metadata,
        }

    def list_sessions(self, request_id_prefix: Optional[str] = None) -> list[str]:
        """List available sessions."""
        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                if request_id_prefix is None or session_dir.name.startswith(request_id_prefix):
                    sessions.append(session_dir.name)
        return sorted(sessions, reverse=True)  # Most recent first