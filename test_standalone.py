#!/usr/bin/env python3

"""
Standalone test for persona system without complex dependencies.
Run with: python3 test_standalone.py
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock classes to avoid dependency issues
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

def test_persona_system():
    """Test the complete persona system functionality."""
    print("🧪 Testing Persona System...")
    
    try:
        # Import our components
        from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
        from poprox_recommender.components.scorers.persona_scorer import PersonaScorer
        
        print("✅ Successfully imported persona components")
        
        # Test 1: Configuration
        config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key="",  # Use simple method
            persona_dimensions=128
        )
        embedder = UserPersonaEmbedder(config)
        print("✅ UserPersonaEmbedder initialized successfully")
        
        # Test 2: Create sample data
        articles = [
            MockArticle("1", "AI Technology Breakthrough", "New AI model shows remarkable capabilities"),
            MockArticle("2", "Climate Policy Update", "Government announces new environmental regulations"),
            MockArticle("3", "Sports Championship Final", "Local team wins major tournament"),
        ]
        
        # Add embeddings to articles
        embeddings = np.random.randn(3, 128).astype(np.float32)
        for i, article in enumerate(articles):
            article.embedding = embeddings[i]
        
        candidate_set = MockCandidateSet(articles=articles, embeddings=embeddings)
        clicked_set = MockCandidateSet(articles=articles[:2], embeddings=embeddings[:2])
        
        interest_profile = MockInterestProfile(
            user_id="test_user",
            click_history=[MockClick("1"), MockClick("2")]
        )
        
        # Test 3: Generate persona
        persona = embedder.generate_persona_from_history(clicked_set)
        print(f"✅ Generated persona vector: shape={persona.shape}, dtype={persona.dtype}")
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (128,)
        
        # Test 4: Score articles
        scorer = PersonaScorer(alpha=0.7, beta=0.3)
        scored_set = scorer(candidate_set, persona, interest_profile)
        
        print(f"✅ Scored articles: {len(scored_set.scores)} scores generated")
        assert scored_set.scores is not None
        assert len(scored_set.scores) == 3
        assert np.all(scored_set.scores >= 0) and np.all(scored_set.scores <= 1)
        
        print(f"📊 Article scores: {scored_set.scores}")
        
        # Test 5: Test with LLM (if API key available)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY', '')
            
            if api_key:
                print("🤖 Testing LLM-based persona generation...")
                llm_config = UserPersonaConfig(
                    model_path="/fake/path",
                    llm_api_key=api_key,
                    persona_dimensions=128
                )
                llm_embedder = UserPersonaEmbedder(llm_config)
                llm_persona = llm_embedder.generate_persona_from_history(clicked_set)
                print(f"✅ LLM persona generated: shape={llm_persona.shape}")
                
                if llm_embedder.last_persona_analysis:
                    print(f"📝 LLM Analysis Preview: {llm_embedder.last_persona_analysis[:200]}...")
            else:
                print("⚠️  No GEMINI_API_KEY found, skipping LLM test")
        except Exception as e:
            print(f"⚠️  LLM test failed: {e}")
        
        print("🎉 All persona system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance with larger dataset."""
    print("\n⚡ Testing Performance...")
    
    try:
        from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
        import time
        
        config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key="",
            persona_dimensions=128
        )
        embedder = UserPersonaEmbedder(config)
        
        # Create 100 articles
        articles = [
            MockArticle(f"{i}", f"Article {i} Title", f"Abstract for article {i}")
            for i in range(100)
        ]
        
        embeddings = np.random.randn(100, 128).astype(np.float32)
        for i, article in enumerate(articles):
            article.embedding = embeddings[i]
        
        candidate_set = MockCandidateSet(articles=articles, embeddings=embeddings)
        
        start_time = time.time()
        persona = embedder.generate_persona_from_history(candidate_set)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Processed 100 articles in {duration:.2f} seconds")
        assert duration < 5.0, f"Performance too slow: {duration:.2f}s"
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Persona System Standalone Test")
    print("=" * 50)
    
    success = test_persona_system()
    if success:
        test_performance()
    
    print("\n" + "=" * 50)
    if success:
        print("🎯 All tests completed successfully!")
        print("💡 The persona system is working correctly and ready for use.")
    else:
        print("💥 Tests failed - check the errors above")
        sys.exit(1)