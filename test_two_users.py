#!/usr/bin/env python3
"""
Production Persona System Test - Two Real Users
Compare how the persona system handles different user behaviors.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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


def categorize_article(title):
    """Categorize articles based on title content."""
    title_lower = title.lower()
    if any(word in title_lower for word in ['economy', 'economic', 'wall street', 'financial', 'market', 'tax', 'labor']):
        return 'business'
    elif any(word in title_lower for word in ['court', 'judge', 'legal', 'justice', 'law', 'supreme court', 'federal']):
        return 'politics'  
    elif any(word in title_lower for word in ['science', 'evolution', 'research', 'reptile', 'ancient', 'measles', 'health', 'epa', 'chemical', 'mars', 'nasa']):
        return 'science'
    elif any(word in title_lower for word in ['international', 'un ', 'gaza', 'colombia', 'ukraine', 'romania', 'portugal', 'uk ', 'eu', 'israel', 'venezuela']):
        return 'international'
    elif any(word in title_lower for word in ['storm', 'tornado', 'weather', 'disaster', 'damage']):
        return 'weather'
    elif any(word in title_lower for word in ['pope', 'church', 'catholic', 'religious']):
        return 'religion'
    elif any(word in title_lower for word in ['sport', 'indy 500', 'qualifying', 'pga', 'golf', 'race', 'tennis', 'bee day']):
        return 'sports'
    elif any(word in title_lower for word in ['tech', 'ai', 'app', 'apple', 'iphone', 'software']):
        return 'technology'
    else:
        return 'general'


def test_user(user_name, clicked_titles, non_clicked_titles, embedder, scorer):
    """Test persona generation for one user."""
    print(f"\n🧠 USER: {user_name}")
    print("=" * 50)
    
    # Create historical articles and clicked articles
    historical_articles = []
    clicked_articles = []
    
    # Add clicked articles to both lists
    for i, title in enumerate(clicked_titles):
        article_id = f"click_{i+1}"
        category = categorize_article(title)
        newsletter_id = f"newsletter_{(i // 4) + 1}"  # Group into newsletters
        
        clicked_articles.append(MockArticle(article_id, title, category=category))
        historical_articles.append(MockArticle(article_id, title, newsletter_id=newsletter_id, category=category))
    
    # Add non-clicked articles to historical list
    for i, title in enumerate(non_clicked_titles):
        article_id = f"ignore_{i+1}"
        category = categorize_article(title)
        newsletter_id = f"newsletter_{(i // 4) + 1}"  # Group into newsletters
        
        historical_articles.append(MockArticle(article_id, title, newsletter_id=newsletter_id, category=category))
    
    # Today's candidates
    candidate_articles = [
        MockArticle("new1", "New archaeological discovery reveals early human migration patterns", "Scientists uncover ancient civilizations", category="science"),
        MockArticle("new2", "Federal Reserve announces policy changes affecting mortgage rates", "Central bank adjusts interest rates", category="business"),
        MockArticle("new3", "Supreme Court hears landmark case on environmental regulations", "Justices decide on EPA authority", category="politics"),
        MockArticle("new4", "International summit addresses global climate change initiatives", "World leaders discuss carbon reduction", category="international"),
        MockArticle("new5", "NFL playoffs see record viewership numbers", "Championship games draw massive audiences", category="sports"),
        MockArticle("new6", "New AI system revolutionizes medical diagnosis", "Machine learning breakthrough in healthcare", category="technology"),
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
        user_id=user_name.lower().replace(' ', '_'),
        click_history=[MockClick(f"click_{i+1}") for i in range(len(clicked_articles))]
    )
    
    # Generate persona
    persona = embedder.run(candidate_set, clicked_set, interest_profile, historical_set)
    
    print(f"🧠 Generated Persona Vector:")
    print(f"   📐 Shape: {persona.shape}")
    print(f"   📊 Stats: mean={np.mean(persona):.4f}, std={np.std(persona):.4f}")
    print(f"   📈 Range: [{np.min(persona):.4f}, {np.max(persona):.4f}]")
    print(f"   🎯 Non-zero elements: {np.count_nonzero(persona)}/{len(persona)}")
    
    # Analyze patterns
    clicked_categories = {}
    ignored_categories = {}
    
    for article in clicked_articles:
        cat = article.category
        clicked_categories[cat] = clicked_categories.get(cat, 0) + 1
        
    for article in historical_articles:
        if article.article_id.startswith('ignore_'):
            cat = article.category
            ignored_categories[cat] = ignored_categories.get(cat, 0) + 1
    
    print(f"\n📊 Engagement Analysis:")
    print(f"   Total: {len(clicked_articles)} clicks, {len([a for a in historical_articles if a.article_id.startswith('ignore_')])} non-clicks")
    
    print(f"   ✅ CLICKED CATEGORIES:")
    for cat, count in sorted(clicked_categories.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(clicked_articles)) * 100
        print(f"      • {cat.title()}: {count} ({pct:.1f}%)")
    
    print(f"   ❌ IGNORED CATEGORIES:")  
    total_ignored = sum(ignored_categories.values())
    for cat, count in sorted(ignored_categories.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_ignored) * 100 if total_ignored > 0 else 0
        print(f"      • {cat.title()}: {count} ({pct:.1f}%)")
    
    # Show LLM analysis if available
    if hasattr(embedder, 'last_persona_analysis') and embedder.last_persona_analysis:
        print(f"\n🤖 LLM Persona Analysis:")
        lines = embedder.last_persona_analysis.split('\n')[:6]  # First 6 lines
        for line in lines:
            if line.strip():
                print(f"   {line}")
        if len(embedder.last_persona_analysis.split('\n')) > 6:
            print("   ... (see full analysis for complete insights)")
    
    # Score articles
    scored_set = scorer(candidate_set, persona, interest_profile)
    
    # Show results
    results = list(zip(candidate_articles, scored_set.scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Personalized Recommendations:")
    for i, (article, score) in enumerate(results, 1):
        print(f"   {i}. {article.title[:60]}... ({article.category}) - {score:.3f}")
    
    return persona, results


def main():
    """Test persona system with two different real users."""
    print("🧠 PERSONA SYSTEM - TWO REAL USERS COMPARISON")
    print("=" * 70)
    
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
        
        # USER 1 DATA - Diverse serious news consumer
        user1_clicked = [
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
        
        user1_non_clicked = [
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
        
        # USER 2 DATA - Tech/Politics focused with interesting variety  
        user2_clicked = [
            "UK's Starmer condemns 'attack on our democracy' after fires at homes linked to him",
            "What the EPA's partial rollback of the 'forever chemical' drinking water rule means",
            "Supreme Court could block Trump's birthright citizenship order but limit nationwide injunctions",
            "Sailing from Oregon to Hawaii after quitting his job turns a man with a cat into social media star",
            "AP Decision Notes: What to expect in Pennsylvania's state primaries",
            "House Republicans include a 10-year ban on US states regulating AI in 'big, beautiful' bill",
            "Still finding trouble in the woods: 'Blair Witch Project' star at center of Maine road dispute",
            "Portugal holds a third general election in 3 years. But the vote might not end the political turmoil",
            "Photo group says it has 'suspended attribution' of historic Vietnam picture because of doubts",
            "Trump budget would cut ocean data and leave boaters, anglers and forecasters scrambling for info",
            "Portugal's election result falls short of ending political instability. Here's what to know",
            "On 'World Bee Day,' the bees did not seem bothered. They should be",
            "Most AAPI adults oppose college funding cuts and student deportations, a new poll finds",
            "What to know about the US Senate's effort to block vehicle-emission rules in California",
            "NASA's Mars Perseverance snaps a selfie as a Martian dust devil blows by",
            "One Tech Tip: These are the apps that can now avoid Apple's in-app payment system"
        ]
        
        user2_non_clicked = [
            "At least 60 people killed by Israeli strikes in Gaza as Israel lets minimal aid in",
            "Apple has had few incentives in the past to start making iPhones in US",
            "JBS shareholders approve US stock listing despite pushback from environmental groups and others",
            "Venezuelan workers at Disney put on leave from jobs after losing protective status",
            "Judge blocks another Trump executive order targeting a major law firm",
            "Blind tennis champion Naqi Rizvi lobbies for sport's awareness and Paralympic inclusion",
            "Federal judge blocks Trump administration from barring foreign student enrollment at Harvard",
            "How ancient reptile footprints are rewriting the history of when animals evolved to live on land",
            "Divisions emerge among House Republicans over how much to cut taxes and Medicaid in Trump's bill",
            "Pope meets Sinner: No. 1 player gives tennis fan Pope Leo XIV racket on Italian Open off day",
            "Harvard joins colleges moving to self-fund some research to offset federal funding cuts",
            "After a judge cut their sentences, the Menendez brothers face a parole board next",
            "Thai officials seize over 200 tons of electronic waste illegally imported from the US",
            "Harvard thought it had a cheap copy of the Magna Carta. It turned out to be extremely rare",
            "Workers are saying 'no' to toxic environments. Here's how to set limits or know it's time to leave",
            "Colombian lawmakers reject president's labor reform referendum",
            "A Texas effort to clarify abortion ban reaches a key vote, but doubts remain",
            "Survivors of clergy sexual abuse turn up calls for reforms from new pope's American hometown",
            "Majority of US states now have laws banning or regulating cellphones in schools, with more to follow",
            "Trial of Maradona's doctors is suspended after judge is accused of authorizing documentary"
        ]
        
        # Test User 1
        user1_persona, user1_results = test_user("Diverse News Consumer", user1_clicked, user1_non_clicked, embedder, scorer)
        
        # Test User 2  
        user2_persona, user2_results = test_user("Tech/Politics Enthusiast", user2_clicked, user2_non_clicked, embedder, scorer)
        
        # Compare personas
        print(f"\n📊 PERSONA COMPARISON")
        print("=" * 50)
        
        persona_similarity = np.dot(user1_persona, user2_persona) / (np.linalg.norm(user1_persona) * np.linalg.norm(user2_persona))
        print(f"🔗 Persona Similarity: {persona_similarity:.3f} (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
        
        print(f"\n🎯 TOP RECOMMENDATION COMPARISON:")
        print(f"User 1 top choice: {user1_results[0][0].title[:50]}... - {user1_results[0][1]:.3f}")
        print(f"User 2 top choice: {user2_results[0][0].title[:50]}... - {user2_results[0][1]:.3f}")
        
        print(f"\n🎉 COMPARISON COMPLETED!")
        print(f"Both users analyzed - you can see how different engagement patterns create different personas!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()