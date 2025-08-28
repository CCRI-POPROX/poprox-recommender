# pyright: basic

import logging
import numpy as np

try:
    from lenskit.pipeline import Component
    from poprox_concepts import CandidateSet, InterestProfile
except ImportError:
    # Fallback for testing without full dependencies
    Component = object
    CandidateSet = object
    InterestProfile = object

logger = logging.getLogger(__name__)


class PersonaScorer(Component):
    """Score candidate articles based on user persona similarity."""
    
    config: None
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        """
        Initialize PersonaScorer.
        
        Args:
            alpha: Weight for persona-content similarity
            beta: Weight for engagement pattern matching
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, 
                 candidate_articles: CandidateSet, 
                 user_persona: np.ndarray,
                 interest_profile: InterestProfile,
                 **kwargs) -> CandidateSet:
        """
        Score candidate articles based on user persona.
        
        Args:
            candidate_articles: Articles to score
            user_persona: User persona vector
            interest_profile: User's interest profile
        
        Returns:
            CandidateSet with computed scores
        """
        if candidate_articles.embeddings is None:
            logger.warning("No article embeddings available for persona scoring")
            return candidate_articles
        
        if user_persona is None or len(user_persona) == 0:
            logger.warning("No user persona available, using uniform scoring")
            scores = np.ones(len(candidate_articles.articles)) * 0.5
        else:
            scores = self._compute_persona_scores(
                candidate_articles.embeddings, 
                user_persona,
                interest_profile
            )
        
        # Create new CandidateSet with scores
        try:
            # Try to create with full CandidateSet if available
            scored_articles = CandidateSet(
                articles=candidate_articles.articles,
                embeddings=candidate_articles.embeddings,
                scores=scores
            )
        except TypeError:
            # Fallback for testing - create copy and add scores
            scored_articles = type(candidate_articles)(
                articles=candidate_articles.articles,
                embeddings=candidate_articles.embeddings,
                scores=scores
            )
        
        return scored_articles
    
    def _compute_persona_scores(self, 
                              article_embeddings: np.ndarray, 
                              user_persona: np.ndarray,
                              interest_profile: InterestProfile) -> np.ndarray:
        """
        Compute similarity scores between user persona and articles.
        
        Args:
            article_embeddings: Article embedding vectors
            user_persona: User persona vector
            interest_profile: User's interest profile for engagement patterns
        
        Returns:
            Array of similarity scores
        """
        # Ensure persona vector has same dimensions as article embeddings
        if len(user_persona) != article_embeddings.shape[1]:
            # Resize persona to match article embedding dimensions
            if len(user_persona) > article_embeddings.shape[1]:
                user_persona = user_persona[:article_embeddings.shape[1]]
            else:
                # Pad with zeros
                user_persona = np.pad(
                    user_persona, 
                    (0, article_embeddings.shape[1] - len(user_persona))
                )
        
        # Normalize vectors
        user_persona_norm = user_persona / (np.linalg.norm(user_persona) + 1e-8)
        article_norms = np.linalg.norm(article_embeddings, axis=1, keepdims=True)
        article_embeddings_norm = article_embeddings / (article_norms + 1e-8)
        
        # Compute cosine similarity
        content_scores = np.dot(article_embeddings_norm, user_persona_norm)
        
        # Compute engagement pattern scores
        engagement_scores = self._compute_engagement_scores(
            article_embeddings, interest_profile
        )
        
        # Combine scores
        final_scores = (self.alpha * content_scores + 
                       self.beta * engagement_scores)
        
        # Ensure scores are in [0, 1] range
        final_scores = np.clip(final_scores, 0.0, 1.0)
        
        return final_scores
    
    def _compute_engagement_scores(self, 
                                 article_embeddings: np.ndarray,
                                 interest_profile: InterestProfile) -> np.ndarray:
        """
        Compute scores based on historical engagement patterns.
        
        Args:
            article_embeddings: Article embedding vectors  
            interest_profile: User's interest profile
        
        Returns:
            Array of engagement-based scores
        """
        if not hasattr(interest_profile, 'click_history') or not interest_profile.click_history:
            return np.ones(len(article_embeddings)) * 0.5
        
        # Simple engagement scoring based on click history recency
        num_articles = len(article_embeddings)
        engagement_scores = np.ones(num_articles) * 0.5
        
        # Recent clicks get higher engagement scores
        recent_click_count = min(len(interest_profile.click_history), 10)
        if recent_click_count > 0:
            # Boost score for articles similar to recently clicked ones
            recency_weight = 0.8 + (recent_click_count / 50.0) * 0.2
            engagement_scores = engagement_scores * recency_weight
        
        return np.clip(engagement_scores, 0.0, 1.0)