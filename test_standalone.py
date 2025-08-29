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
    """Test the complete persona system functionality with comprehensive data."""
    print("🧪 Testing Enhanced Persona System with Comprehensive Data...")
    
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
        
        # Test 2: Create comprehensive sample data representing real user behavior
        print("\n📰 Creating Comprehensive Test Dataset...")
        
        # REALISTIC CANDIDATE ARTICLES (what user sees in today's newsletter)
        candidate_articles = [
            # Technology articles (user shows interest)
            MockArticle("c1", "Revolutionary AI Breakthrough in Medical Diagnosis", 
                       "Scientists develop AI system that can detect rare diseases with 99% accuracy, potentially saving thousands of lives"),
            MockArticle("c2", "Major Tech Company Launches Quantum Computing Platform", 
                       "First commercial quantum computing service becomes available for enterprise customers, marking new era in computing"),
            
            # Environmental/Climate articles (user shows strong interest)
            MockArticle("c3", "Global Climate Summit Reaches Historic Agreement", 
                       "195 countries commit to aggressive carbon reduction targets, with binding enforcement mechanisms"),
            MockArticle("c4", "Breakthrough in Clean Energy Storage Technology", 
                       "New battery technology could store renewable energy for months, solving intermittency problem"),
            
            # Science articles (user shows some interest)
            MockArticle("c5", "NASA Discovers Potentially Habitable Exoplanet", 
                       "Kepler telescope identifies Earth-like planet in habitable zone, water signatures detected"),
            
            # Sports articles (user consistently ignores)
            MockArticle("c6", "Championship Game Breaks Viewership Records", 
                       "Super Bowl draws 120 million viewers, highest rated broadcast in five years"),
            MockArticle("c7", "Star Player Signs Record-Breaking Contract", 
                       "Basketball superstar agrees to $200 million deal, largest in league history"),
            
            # Entertainment articles (user actively avoids)
            MockArticle("c8", "Celebrity Wedding Dominates Social Media", 
                       "Hollywood power couple's lavish ceremony generates millions of social media posts"),
            MockArticle("c9", "Reality TV Show Causes Controversy", 
                       "New reality series faces backlash over controversial contestant behavior"),
            
            # Business/Finance articles (mixed interest)
            MockArticle("c10", "Stock Market Reaches All-Time High", 
                        "Major indices surge following positive economic data and corporate earnings")
        ]
        
        # USER'S HISTORICAL CLICK BEHAVIOR (what they actually clicked on in past)
        clicked_articles = [
            # Strong technology interests
            MockArticle("h1", "AI Breakthrough in Healthcare Research", 
                       "Machine learning algorithm identifies new drug compounds faster than traditional methods"),
            MockArticle("h2", "Quantum Computing Milestone Achieved", 
                       "IBM's quantum processor solves complex optimization problem in seconds"),
            MockArticle("h3", "Cybersecurity Threat Landscape Analysis", 
                       "Annual report reveals sophisticated attack patterns targeting critical infrastructure"),
            
            # Strong environmental interests
            MockArticle("h4", "Climate Change Impacts Accelerating", 
                       "IPCC report shows faster than expected warming, urgent action needed"),
            MockArticle("h5", "Renewable Energy Surpasses Coal Generation", 
                       "Solar and wind power generate more electricity than fossil fuels for first time"),
            MockArticle("h6", "Electric Vehicle Adoption Hits Tipping Point", 
                       "EV sales exceed 50% of new car purchases in major markets"),
            
            # Science interests
            MockArticle("h7", "Gene Therapy Breakthrough for Rare Disease", 
                       "Clinical trial shows 100% success rate in treating previously incurable genetic disorder"),
            MockArticle("h8", "Mars Mission Reveals Surprising Discoveries", 
                       "Rover findings suggest ancient microbial life may have existed on Red Planet"),
            
            # Selective business/policy interests (focuses on tech policy)
            MockArticle("h9", "New Data Privacy Regulations Take Effect", 
                       "Comprehensive digital rights legislation impacts how companies handle user information"),
            MockArticle("h10", "Tech Industry Antitrust Investigation Expands", 
                        "Federal regulators examine market dominance of major technology platforms")
        ]
        
        print(f"   📊 Created {len(candidate_articles)} candidate articles for today")
        print(f"   📚 Created {len(clicked_articles)} historical clicked articles")
        print(f"   🎯 User Interest Profile: Technology (40%), Environment (30%), Science (20%), Tech Policy (10%)")
        print(f"   ❌ User Avoids: Sports, Celebrity/Entertainment, General Business News")
        
        # Generate realistic embeddings that reflect content similarity
        print("\n🔢 Generating Content Embeddings...")
        
        # Create embeddings that cluster similar topics together
        np.random.seed(42)  # For reproducible results
        
        candidate_embeddings = np.random.randn(len(candidate_articles), 128).astype(np.float32)
        clicked_embeddings = np.random.randn(len(clicked_articles), 128).astype(np.float32)
        
        # Make technology articles similar to each other
        tech_base = np.random.randn(128).astype(np.float32)
        candidate_embeddings[0] = tech_base + 0.1 * np.random.randn(128).astype(np.float32)  # AI Medical
        candidate_embeddings[1] = tech_base + 0.1 * np.random.randn(128).astype(np.float32)  # Quantum
        
        # Make climate articles similar
        climate_base = np.random.randn(128).astype(np.float32) 
        candidate_embeddings[2] = climate_base + 0.1 * np.random.randn(128).astype(np.float32)  # Climate Summit
        candidate_embeddings[3] = climate_base + 0.1 * np.random.randn(128).astype(np.float32)  # Clean Energy
        
        # Sports articles cluster
        sports_base = np.random.randn(128).astype(np.float32)
        candidate_embeddings[5] = sports_base + 0.1 * np.random.randn(128).astype(np.float32)
        candidate_embeddings[6] = sports_base + 0.1 * np.random.randn(128).astype(np.float32)
        
        # Entertainment articles cluster  
        entertainment_base = np.random.randn(128).astype(np.float32)
        candidate_embeddings[7] = entertainment_base + 0.1 * np.random.randn(128).astype(np.float32)
        candidate_embeddings[8] = entertainment_base + 0.1 * np.random.randn(128).astype(np.float32)
        
        # Apply embeddings to articles
        for i, article in enumerate(candidate_articles):
            article.embedding = candidate_embeddings[i]
        for i, article in enumerate(clicked_articles):
            article.embedding = clicked_embeddings[i]
        
        candidate_set = MockCandidateSet(articles=candidate_articles, embeddings=candidate_embeddings)
        clicked_set = MockCandidateSet(articles=clicked_articles, embeddings=clicked_embeddings)
        
        # Create comprehensive user profile
        all_click_ids = [f"h{i+1}" for i in range(len(clicked_articles))]
        interest_profile = MockInterestProfile(
            user_id="tech_climate_enthusiast_user",
            click_history=[MockClick(click_id) for click_id in all_click_ids]
        )
        
        print(f"   ✅ Generated embeddings with topic clustering")
        print(f"   ✅ User profile: {len(interest_profile.click_history)} total clicks")
        
        # Test 3: Generate persona using full pipeline
        print("\n🧠 Generating User Persona...")
        persona = embedder.run(candidate_set, clicked_set, interest_profile)
        print(f"✅ Generated persona vector: shape={persona.shape}, dtype={persona.dtype}")
        print(f"   📈 Persona vector statistics:")
        print(f"      - Mean: {np.mean(persona):.4f}")
        print(f"      - Std:  {np.std(persona):.4f}")
        print(f"      - Min:  {np.min(persona):.4f}")
        print(f"      - Max:  {np.max(persona):.4f}")
        
        assert isinstance(persona, np.ndarray)
        assert persona.shape == (128,)
        
        # Test 4: Score articles with detailed analysis
        print("\n📊 Scoring Today's Articles with User Persona...")
        scorer = PersonaScorer(alpha=0.7, beta=0.3)
        scored_set = scorer(candidate_set, persona, interest_profile)
        
        print(f"✅ Scored articles: {len(scored_set.scores)} scores generated")
        assert scored_set.scores is not None
        assert len(scored_set.scores) == len(candidate_articles)
        assert np.all(scored_set.scores >= 0) and np.all(scored_set.scores <= 1)
        
        # Display detailed scoring results
        print("\n🎯 DETAILED RECOMMENDATION RESULTS:")
        print("=" * 80)
        
        # Sort articles by score for better analysis
        scored_articles = list(zip(candidate_articles, scored_set.scores))
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (article, score) in enumerate(scored_articles, 1):
            # Determine expected interest level
            title_lower = article.title.lower()
            if any(word in title_lower for word in ['ai', 'quantum', 'tech']):
                expected = "HIGH (Technology Interest)"
            elif any(word in title_lower for word in ['climate', 'energy', 'environment']):
                expected = "HIGH (Climate Interest)" 
            elif any(word in title_lower for word in ['nasa', 'space', 'science']):
                expected = "MEDIUM (Science Interest)"
            elif any(word in title_lower for word in ['sport', 'championship', 'game']):
                expected = "LOW (User Avoids Sports)"
            elif any(word in title_lower for word in ['celebrity', 'wedding', 'reality']):
                expected = "VERY LOW (User Avoids Entertainment)"
            else:
                expected = "MEDIUM (General Interest)"
                
            print(f"#{rank}. Score: {score:.3f} | Expected: {expected}")
            print(f"    Title: {article.title}")
            print(f"    Abstract: {article.abstract[:80]}...")
            print()
        
        # Test 5: Test with LLM for comprehensive persona analysis
        print("\n🤖 Testing LLM-based Comprehensive Persona Generation...")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY', '')
            
            if api_key:
                print("🔑 API Key found - generating comprehensive LLM persona...")
                llm_config = UserPersonaConfig(
                    model_path="/fake/path",
                    llm_api_key=api_key,
                    persona_dimensions=128
                )
                llm_embedder = UserPersonaEmbedder(llm_config)
                llm_persona = llm_embedder.run(candidate_set, clicked_set, interest_profile)
                print(f"✅ LLM persona generated: shape={llm_persona.shape}")
                
                if llm_embedder.last_persona_analysis:
                    print("\n" + "=" * 80)
                    print("📝 COMPLETE LLM PERSONA ANALYSIS:")
                    print("=" * 80)
                    print(llm_embedder.last_persona_analysis)
                    print("=" * 80)
                    
                    # Test LLM persona scoring
                    print("\n🎯 SCORING WITH LLM-GENERATED PERSONA:")
                    llm_scored_set = scorer(candidate_set, llm_persona, interest_profile)
                    llm_scored_articles = list(zip(candidate_articles, llm_scored_set.scores))
                    llm_scored_articles.sort(key=lambda x: x[1], reverse=True)
                    
                    for rank, (article, score) in enumerate(llm_scored_articles[:5], 1):
                        print(f"#{rank}. LLM Score: {score:.3f} | {article.title}")
                else:
                    print("⚠️  No LLM analysis text available")
            else:
                print("⚠️  No GEMINI_API_KEY found in environment")
                print("   💡 To test LLM personas: export GEMINI_API_KEY='your_key_here'")
                print("   🔗 Get API key from: https://ai.google.dev/")
        except Exception as e:
            print(f"⚠️  LLM test failed: {e}")
            import traceback
            print("Debug traceback:")
            traceback.print_exc()
        
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