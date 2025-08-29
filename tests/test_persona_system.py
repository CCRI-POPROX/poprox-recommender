#!/usr/bin/env python3
"""
Production Persona System Test
Simple test to verify the persona system works correctly.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock classes
@dataclass 
class MockArticle:
    article_id: str
    title: str
    abstract: str = ""
    embedding: Optional[np.ndarray] = None
    newsletter_id: str = "newsletter_1"
    category: str = "general"

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
    """Test the persona system with two different real users."""
    print("🧠 Persona System Production Test - Two Real Users")
    print("=" * 60)
    
    try:
        from poprox_recommender.components.embedders.user_persona import UserPersonaEmbedder, UserPersonaConfig
        from poprox_recommender.components.scorers.persona_scorer import PersonaScorer
        
        print("✅ Components imported")
        
        # Configuration
        config = UserPersonaConfig(
            model_path="/fake/path",
            llm_api_key=os.getenv('GEMINI_API_KEY', ''),
            persona_dimensions=128,
            max_clicks_per_user=50,
            max_non_clicks_per_user=200
        )
        embedder = UserPersonaEmbedder(config)
        scorer = PersonaScorer()
        
        print("✅ System initialized")
        
        # Test both users to show different persona patterns
        user_choice = input("Choose user to analyze (1 or 2, or 'both' for comparison): ").strip().lower()
        
        def test_user(user_num, clicked_titles, non_clicked_titles, user_description):
            print(f"\n📰 Analyzing {user_description}")
            print("=" * 50)
        
        # USER 1 DATA - Diverse serious news consumer
        user1_clicked_titles = [
            "How ancient reptile footprints are rewriting the history of when animals evolved to live on land",
            "In their words: What judges have said about birthright citizenship and nationwide injunctions", 
            "UK becomes fastest-growing G7 economy after strong first quarter",
            "What the EPA's partial rollback of the 'forever chemical' drinking water rule means",
            "Colombian lawmakers reject president's labor reform referendum",
            "What to know about the Menendez brothers' bid for freedom",
            "AP Decision Notes: What to expect in Pennsylvania's state primaries",
            "DeSantis signs a bill making Florida the 2nd state to ban fluoride from its water system",
            "UN aid chief defends using 'genocide' in Gaza remarks to the Security Council that Israel rejects",
            "Wall Street drifts back within 4% of its record after the S&P 500 notches a 4th straight gain",
            "Latin America's leftist leaders remember Uruguay's 'Pepe' Mujica as generous, charismatic leader",
            "Severe thunderstorms down trees, knock out power to thousands across parts of Great Lakes region",
            "A Texas suburb that saw its population jump by a third is the fastest-growing city in the US",
            "Texas' measles outbreak is starting to slow. The US case count climbs slightly to 1,024 cases",
            "Conservatives block Trump's big tax breaks bill in a stunning setback",
            "Federal judge strikes down workplace protections for transgender workers",
            "UN agency, Rohingya refugees allege Indian authorities cast dozens of them into the sea near Myanmar",
            "Justice Department deal ends a ban on an aftermarket trigger. Gun control advocates are alarmed",
            "Pope Leo XIV vows to work for unity so Catholic Church becomes a sign of peace in the world"
        ]
        
        # Real user's non-clicked articles (from newsletters they engaged with)
        real_non_clicked_titles = [
            "Romanians vote in a tense presidential runoff that pits nationalist against pro-EU centrist",
            "Residents dig out from tornado damage after storms kill 27 in Kentucky, Missouri and Virginia",
            "Federal officials launch investigation into Mexican tall ship that struck Brooklyn Bridge",
            "Portugal holds a third general election in 3 years. But the vote might not end the political turmoil",
            "Pope Leo XIV vows to work for unity so Catholic Church becomes a symbol of peace in the world",
            "Trump's clash with the courts raises prospect of showdown over separation of powers", 
            "Palou and Penske set pace in 1st stage of Indy 500 qualifying. Andretti in danger of missing race",
            "A second suspect has been arrested over fires targeting UK Prime Minister Keir Starmer's properties",
            "What to know about the Menendez brothers' bid for freedom",
            "What to know about the Menendez brothers' resentencing plea",
            "Supreme Court could block Trump's birthright citizenship order but limit nationwide injunctions",
            "Pro-EU centrist wins Romania's tense presidential race over hard-right nationalist",
            "The UK and the EU are to seal new deals and renew ties 5 years after Brexit",
            "Storms and tornadoes across central US kill dozens and damage homes",
            "Appeals court allows Trump's anti-union order to take effect",
            "More storms take aim at central US, where many are digging out from tornado damage",
            "Federal officials investigating Mexican tall ship's crash into Brooklyn Bridge",
            "Netanyahu says he'll allow some aid into Gaza under pressure, but none appears to have entered",
            "Residents dig out from tornado damage after storms kill 28 in Kentucky, Missouri and Virginia",
            "Report: McIlroy's driver deemed nonconforming ahead of PGA Championship"
        ]
        
        # Categorize articles based on content analysis
        def categorize_article(title):
            title_lower = title.lower()
            if any(word in title_lower for word in ['economy', 'economic', 'wall street', 'financial', 'market', 'tax', 'labor']):
                return 'business'
            elif any(word in title_lower for word in ['court', 'judge', 'legal', 'justice', 'law', 'supreme court', 'federal']):
                return 'politics'  
            elif any(word in title_lower for word in ['science', 'evolution', 'research', 'reptile', 'ancient', 'measles', 'health', 'epa', 'chemical']):
                return 'science'
            elif any(word in title_lower for word in ['international', 'un ', 'gaza', 'colombia', 'ukraine', 'romania', 'portugal', 'uk ', 'eu']):
                return 'international'
            elif any(word in title_lower for word in ['storm', 'tornado', 'weather', 'disaster', 'damage']):
                return 'weather'
            elif any(word in title_lower for word in ['pope', 'church', 'catholic', 'religious']):
                return 'religion'
            elif any(word in title_lower for word in ['sport', 'indy 500', 'qualifying', 'pga', 'golf', 'race']):
                return 'sports'
            else:
                return 'general'
        
        # Create historical articles (simulate newsletters)
        historical_articles = []
        clicked_articles = []
        
        # Add clicked articles to both lists
        for i, title in enumerate(real_clicked_titles):
            article_id = f"click_{i+1}"
            category = categorize_article(title)
            newsletter_id = f"newsletter_{(i // 4) + 1}"  # Group into newsletters
            
            clicked_articles.append(MockArticle(article_id, title, category=category))
            historical_articles.append(MockArticle(article_id, title, newsletter_id=newsletter_id, category=category))
        
        # Add non-clicked articles to historical list
        for i, title in enumerate(real_non_clicked_titles):
            article_id = f"ignore_{i+1}"
            category = categorize_article(title)
            newsletter_id = f"newsletter_{(i // 4) + 1}"  # Group into newsletters
            
            historical_articles.append(MockArticle(article_id, title, newsletter_id=newsletter_id, category=category))
        
        # Today's realistic candidates based on user's interests
        candidate_articles = [
            MockArticle("new1", "New archaeological discovery reveals early human migration patterns", "Scientists uncover evidence of ancient civilizations", category="science"),
            MockArticle("new2", "Federal Reserve announces policy changes affecting mortgage rates", "Central bank adjusts interest rates amid economic uncertainty", category="business"),
            MockArticle("new3", "Supreme Court hears landmark case on environmental regulations", "Justices to decide on EPA authority over clean water standards", category="politics"),
            MockArticle("new4", "International summit addresses global climate change initiatives", "World leaders gather to discuss carbon reduction strategies", category="international"),
            MockArticle("new5", "NFL playoffs see record viewership numbers", "Championship games draw massive television audiences", category="sports"),
            MockArticle("new6", "Severe weather warnings issued across multiple states", "Meteorologists predict dangerous storm systems moving east", category="weather"),
        ]
        
        # Add embeddings
        for articles in [historical_articles, clicked_articles, candidate_articles]:
            for article in articles:
                article.embedding = np.random.randn(128).astype(np.float32)
        
        # Create data structures
        historical_set = MockCandidateSet(
            articles=historical_articles,
            embeddings=np.array([a.embedding for a in historical_articles])
        )
        clicked_set = MockCandidateSet(
            articles=clicked_articles,
            embeddings=np.array([a.embedding for a in clicked_articles])
        )
        candidate_set = MockCandidateSet(
            articles=candidate_articles,
            embeddings=np.array([a.embedding for a in candidate_articles])
        )
        
        interest_profile = MockInterestProfile(
            user_id="test_user",
            click_history=[MockClick(f"click_{i}") for i in range(1, 4)]
        )
        
        print("✅ Test data created")
        
        # Generate persona
        persona = embedder.run(candidate_set, clicked_set, interest_profile, historical_set)
        assert isinstance(persona, np.ndarray) and persona.shape == (128,)
        print("✅ Persona generated")
        
        # Show persona details
        print(f"\n🧠 Generated Persona Vector:")
        print(f"   📐 Shape: {persona.shape}")
        print(f"   📊 Stats: mean={np.mean(persona):.4f}, std={np.std(persona):.4f}")
        print(f"   📈 Range: [{np.min(persona):.4f}, {np.max(persona):.4f}]")
        print(f"   🎯 Non-zero elements: {np.count_nonzero(persona)}/{len(persona)}")
        
        # Show LLM analysis if available
        if hasattr(embedder, 'last_persona_analysis') and embedder.last_persona_analysis:
            print(f"\n🤖 LLM Persona Analysis:")
            lines = embedder.last_persona_analysis.split('\n')[:8]  # First 8 lines
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            if len(embedder.last_persona_analysis.split('\n')) > 8:
                print("   ... (truncated)")
        else:
            # Analyze real user patterns
            clicked_categories = {}
            ignored_categories = {}
            
            for article in clicked_articles:
                cat = article.category
                clicked_categories[cat] = clicked_categories.get(cat, 0) + 1
                
            for article in historical_articles:
                if article.article_id.startswith('ignore_'):
                    cat = article.category
                    ignored_categories[cat] = ignored_categories.get(cat, 0) + 1
            
            print(f"\n💡 Real User Persona Analysis:")
            print(f"   📊 Total engagement: {len(clicked_articles)} clicks, {len([a for a in historical_articles if a.article_id.startswith('ignore_')])} non-clicks")
            
            print(f"   ✅ CLICKED CATEGORIES:")
            for cat, count in sorted(clicked_categories.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(clicked_articles)) * 100
                print(f"      • {cat.title()}: {count} articles ({pct:.1f}%)")
            
            print(f"   ❌ IGNORED CATEGORIES:")  
            total_ignored = sum(ignored_categories.values())
            for cat, count in sorted(ignored_categories.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_ignored) * 100 if total_ignored > 0 else 0
                print(f"      • {cat.title()}: {count} articles ({pct:.1f}%)")
                
            print(f"   🧠 Pattern: User engages with diverse serious news, avoids sports/weather disasters")
        
        # Score articles
        scored_set = scorer(candidate_set, persona, interest_profile)
        assert len(scored_set.scores) == 6
        print("✅ Articles scored")
        
        # Check results
        results = list(zip(candidate_articles, scored_set.scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 Results:")
        for i, (article, score) in enumerate(results, 1):
            print(f"   {i}. {article.title} ({article.category}) - {score:.3f}")
        
        print("\n🎉 Production test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_persona_system()