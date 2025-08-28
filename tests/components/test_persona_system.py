# pyright: basic

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Optional

# Mock poprox_concepts since it's not available in test environment
@dataclass 
class MockArticle:
    article_id: str
    title: str
    abstract: str = ""
    embedding: Optional[np.ndarray] = None

@dataclass
class MockCandidateSet:
    articles: List[MockArticle]
    embeddings: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

@dataclass 
class MockClick:
    article_id: str

@dataclass
class MockInterestProfile:
    user_id: str
    click_history: List[MockClick]
    embedding: Optional[np.ndarray] = None

# Mock the imports
with patch.dict('sys.modules', {
    'poprox_concepts': Mock(),
    'lenskit.pipeline': Mock(),
    'lenskit': Mock(),
    'google.generativeai': Mock(),
    'sentence_transformers': Mock(),
}):
    from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
    from poprox_recommender.components.scorers.persona_scorer import PersonaScorer


class TestUserPersonaEmbedder:
    """Test cases for UserPersonaEmbedder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key="fake_key",
            max_history_length=50,
            persona_dimensions=128
        )
        
        # Create sample articles
        self.sample_articles = [
            MockArticle("1", "Tech Company Releases New AI Model", "Major tech company announces breakthrough AI"),
            MockArticle("2", "Climate Change Policy Updates", "Government announces new environmental policies"),
            MockArticle("3", "Sports Championship Finals", "Final game of the season attracts millions"),
            MockArticle("4", "Economic Market Analysis", "Stock market shows unexpected trends"),
            MockArticle("5", "Healthcare Innovation Breakthrough", "New medical technology saves lives")
        ]
        
        # Create embeddings for articles
        self.sample_embeddings = np.random.randn(5, 128).astype(np.float32)
        for i, article in enumerate(self.sample_articles):
            article.embedding = self.sample_embeddings[i]
    
    def test_initialization(self):
        """Test UserPersonaEmbedder initialization."""
        embedder = UserPersonaEmbedder(self.config)
        assert embedder.config == self.config
        assert embedder.persona_cache == {}
        assert embedder.last_persona_analysis == ""
    
    def test_environment_api_key_loading(self):
        """Test API key loading from environment."""
        config_no_key = UserPersonaConfig(model_path="/fake/path")
        
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'env_api_key'}):
            embedder = UserPersonaEmbedder(config_no_key)
            assert embedder.config.llm_api_key == 'env_api_key'
    
    def test_simple_persona_generation(self):
        """Test simple persona generation with article embeddings."""
        embedder = UserPersonaEmbedder(self.config)
        embedder.config.llm_api_key = ""  # Force simple method
        embedder.config.persona_model_path = ""
        
        candidate_set = MockCandidateSet(
            articles=self.sample_articles[:3],
            embeddings=self.sample_embeddings[:3]
        )
        
        persona = embedder.generate_persona_from_history(candidate_set)
        
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (self.config.persona_dimensions,)
        assert not np.allclose(persona, 0)  # Should not be all zeros
    
    def test_empty_article_history(self):
        """Test persona generation with empty article history."""
        embedder = UserPersonaEmbedder(self.config)
        candidate_set = MockCandidateSet(articles=[])
        
        persona = embedder.generate_persona_from_history(candidate_set)
        
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (self.config.persona_dimensions,)
        assert np.allclose(persona, 0)  # Should be all zeros for empty history
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_llm_persona_generation(self, mock_model_class, mock_configure):
        """Test LLM-based persona generation."""
        # Mock the LLM response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "User shows strong interest in technology and AI, moderate interest in climate policy."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        embedder = UserPersonaEmbedder(self.config)
        candidate_set = MockCandidateSet(
            articles=self.sample_articles[:3],
            embeddings=self.sample_embeddings[:3]
        )
        
        persona = embedder.generate_persona_from_history(candidate_set)
        
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (self.config.persona_dimensions,)
        mock_configure.assert_called_once_with(api_key="fake_key")
        mock_model.generate_content.assert_called_once()
        
        # Check that comprehensive prompt was used
        call_args = mock_model.generate_content.call_args[0][0]
        assert "PRIMARY INTEREST AREAS" in call_args
        assert "DISENGAGEMENT ANALYSIS" in call_args
    
    def test_llm_fallback_on_error(self):
        """Test fallback to simple method when LLM fails."""
        embedder = UserPersonaEmbedder(self.config)
        
        # Mock LLM to raise an exception
        with patch('google.generativeai.configure', side_effect=Exception("API Error")):
            candidate_set = MockCandidateSet(
                articles=self.sample_articles[:2],
                embeddings=self.sample_embeddings[:2]
            )
            
            persona = embedder.generate_persona_from_history(candidate_set)
            
            assert isinstance(persona, np.ndarray)
            assert persona.shape == (self.config.persona_dimensions,)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_pretrained_model_generation(self, mock_transformer_class):
        """Test pre-trained model persona generation."""
        # Mock sentence transformer
        mock_model = Mock()
        mock_embeddings = np.random.randn(3, 384).astype(np.float32)
        mock_model.encode.return_value = mock_embeddings
        mock_transformer_class.return_value = mock_model
        
        # Configure for pre-trained model
        config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key="",
            persona_model_path="all-MiniLM-L6-v2",
            persona_dimensions=128
        )
        
        embedder = UserPersonaEmbedder(config)
        candidate_set = MockCandidateSet(
            articles=self.sample_articles[:3],
            embeddings=self.sample_embeddings[:3]
        )
        
        persona = embedder.generate_persona_from_history(candidate_set)
        
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (config.persona_dimensions,)
        mock_model.encode.assert_called_once()
    
    def test_text_to_vector_conversion(self):
        """Test text to vector conversion."""
        embedder = UserPersonaEmbedder(self.config)
        
        text = "User interested in technology and sports"
        vector = embedder._text_to_vector(text)
        
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (self.config.persona_dimensions,)
        assert vector.dtype == np.float32
        assert np.all(vector >= 0) and np.all(vector <= 1)  # Should be normalized
    
    def test_persona_caching(self):
        """Test persona caching functionality."""
        embedder = UserPersonaEmbedder(self.config)
        embedder.config.llm_api_key = ""  # Force simple method
        
        candidate_set = MockCandidateSet(
            articles=self.sample_articles[:2],
            embeddings=self.sample_embeddings[:2]
        )
        
        clicked_set = MockCandidateSet(
            articles=self.sample_articles[:2],
            embeddings=self.sample_embeddings[:2]
        )
        
        interest_profile = MockInterestProfile(
            user_id="user_123",
            click_history=[MockClick("1"), MockClick("2")]
        )
        
        persona = embedder.run(candidate_set, clicked_set, interest_profile)
        
        # Check that persona is cached
        assert "user_123" in embedder.persona_cache
        assert np.array_equal(embedder.persona_cache["user_123"], persona)


class TestPersonaScorer:
    """Test cases for PersonaScorer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = PersonaScorer(alpha=0.7, beta=0.3)
        
        # Create sample data
        self.sample_articles = [
            MockArticle("1", "Tech Innovation News"),
            MockArticle("2", "Sports Update"),
            MockArticle("3", "Climate Policy")
        ]
        
        self.article_embeddings = np.random.randn(3, 128).astype(np.float32)
        self.user_persona = np.random.randn(128).astype(np.float32)
        
        self.candidate_set = MockCandidateSet(
            articles=self.sample_articles,
            embeddings=self.article_embeddings
        )
        
        self.interest_profile = MockInterestProfile(
            user_id="test_user",
            click_history=[MockClick("1"), MockClick("2")]
        )
    
    def test_initialization(self):
        """Test PersonaScorer initialization."""
        scorer = PersonaScorer(alpha=0.6, beta=0.4)
        assert scorer.alpha == 0.6
        assert scorer.beta == 0.4
    
    def test_scoring_with_valid_data(self):
        """Test scoring with valid persona and embeddings."""
        scored_set = self.scorer(
            self.candidate_set,
            self.user_persona,
            self.interest_profile
        )
        
        assert isinstance(scored_set, MockCandidateSet)
        assert scored_set.scores is not None
        assert len(scored_set.scores) == len(self.sample_articles)
        assert np.all(scored_set.scores >= 0) and np.all(scored_set.scores <= 1)
    
    def test_scoring_without_embeddings(self):
        """Test scoring when article embeddings are missing."""
        candidate_set_no_embeddings = MockCandidateSet(
            articles=self.sample_articles,
            embeddings=None
        )
        
        scored_set = self.scorer(
            candidate_set_no_embeddings,
            self.user_persona,
            self.interest_profile
        )
        
        # Should return original candidate set
        assert scored_set == candidate_set_no_embeddings
    
    def test_scoring_without_persona(self):
        """Test scoring when user persona is missing."""
        scored_set = self.scorer(
            self.candidate_set,
            None,
            self.interest_profile
        )
        
        assert isinstance(scored_set, MockCandidateSet)
        assert scored_set.scores is not None
        assert len(scored_set.scores) == len(self.sample_articles)
        # Should use uniform scoring
        assert np.allclose(scored_set.scores, 0.5)
    
    def test_persona_dimension_mismatch(self):
        """Test handling of persona-embedding dimension mismatch."""
        # Create persona with different dimensions
        mismatched_persona = np.random.randn(64).astype(np.float32)
        
        scored_set = self.scorer(
            self.candidate_set,
            mismatched_persona,
            self.interest_profile
        )
        
        assert isinstance(scored_set, MockCandidateSet)
        assert scored_set.scores is not None
        assert len(scored_set.scores) == len(self.sample_articles)
    
    def test_engagement_scoring(self):
        """Test engagement pattern scoring."""
        # Test with different click history lengths
        profile_many_clicks = MockInterestProfile(
            user_id="active_user",
            click_history=[MockClick(f"{i}") for i in range(20)]
        )
        
        profile_few_clicks = MockInterestProfile(
            user_id="inactive_user", 
            click_history=[MockClick("1")]
        )
        
        engagement_scores_many = self.scorer._compute_engagement_scores(
            self.article_embeddings, profile_many_clicks
        )
        
        engagement_scores_few = self.scorer._compute_engagement_scores(
            self.article_embeddings, profile_few_clicks
        )
        
        assert len(engagement_scores_many) == len(self.article_embeddings)
        assert len(engagement_scores_few) == len(self.article_embeddings)
        
        # Users with more clicks should generally have higher engagement scores
        assert np.mean(engagement_scores_many) >= np.mean(engagement_scores_few)


class TestPersonaSystemIntegration:
    """Integration tests for the complete persona system."""
    
    def test_end_to_end_persona_workflow(self):
        """Test complete workflow from persona generation to scoring."""
        # Setup
        config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key="",  # Force simple method
            persona_dimensions=128
        )
        
        embedder = UserPersonaEmbedder(config)
        scorer = PersonaScorer()
        
        # Create sample data
        articles = [
            MockArticle("1", "AI Technology Breakthrough", "New AI model released"),
            MockArticle("2", "Climate Summit Results", "Global climate agreement reached"),
            MockArticle("3", "Sports Championship", "Team wins major tournament")
        ]
        
        embeddings = np.random.randn(3, 128).astype(np.float32)
        for i, article in enumerate(articles):
            article.embedding = embeddings[i]
        
        clicked_set = MockCandidateSet(articles=articles[:2], embeddings=embeddings[:2])
        candidate_set = MockCandidateSet(articles=articles, embeddings=embeddings)
        
        interest_profile = MockInterestProfile(
            user_id="test_user",
            click_history=[MockClick("1"), MockClick("2")]
        )
        
        # Generate persona
        persona = embedder.generate_persona_from_history(clicked_set)
        
        # Score articles
        scored_set = scorer(candidate_set, persona, interest_profile)
        
        # Verify results
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (128,)
        assert scored_set.scores is not None
        assert len(scored_set.scores) == 3
        assert np.all(scored_set.scores >= 0) and np.all(scored_set.scores <= 1)


# Pytest fixtures and test data
@pytest.fixture
def sample_articles():
    """Fixture providing sample articles for testing."""
    return [
        MockArticle("1", "Breaking: Tech Company IPO", "Major technology company goes public"),
        MockArticle("2", "Climate Change Research", "New study reveals climate trends"),
        MockArticle("3", "Olympic Games Update", "Athletes compete in summer games"),
        MockArticle("4", "Economic Policy Analysis", "Government announces fiscal policy"),
        MockArticle("5", "Medical Research Breakthrough", "Scientists discover new treatment")
    ]


@pytest.fixture
def sample_user_profile():
    """Fixture providing sample user profile for testing."""
    return MockInterestProfile(
        user_id="sample_user",
        click_history=[
            MockClick("1"),
            MockClick("2"), 
            MockClick("5")
        ]
    )


# Performance test
def test_persona_generation_performance():
    """Test that persona generation completes within reasonable time."""
    import time
    
    config = UserPersonaConfig(
        model_path="/fake/path",
        llm_api_key="",
        persona_dimensions=128
    )
    
    embedder = UserPersonaEmbedder(config)
    
    # Create large article set
    articles = [
        MockArticle(f"{i}", f"Article {i}", f"Abstract {i}")
        for i in range(100)
    ]
    
    embeddings = np.random.randn(100, 128).astype(np.float32)
    for i, article in enumerate(articles):
        article.embedding = embeddings[i]
    
    candidate_set = MockCandidateSet(articles=articles, embeddings=embeddings)
    
    start_time = time.time()
    persona = embedder.generate_persona_from_history(candidate_set)
    end_time = time.time()
    
    # Should complete within 5 seconds for 100 articles
    assert end_time - start_time < 5.0
    assert isinstance(persona, np.ndarray)
    assert persona.shape == (128,)


if __name__ == "__main__":
    pytest.main([__file__])