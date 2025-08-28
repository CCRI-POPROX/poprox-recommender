#!/usr/bin/env python3
"""
Standalone demo of the persona-based news recommender system.
This script demonstrates the persona generation logic without requiring
the full poprox-concepts and lenskit dependencies.
"""

import os
import numpy as np
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class Article:
    """Simplified article representation."""
    article_id: str
    title: str
    abstract: str = ""
    embedding: Optional[np.ndarray] = None

@dataclass
class CandidateSet:
    """Set of candidate articles."""
    articles: List[Article]
    embeddings: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

@dataclass 
class Click:
    """User click representation."""
    article_id: str

@dataclass
class InterestProfile:
    """User interest profile."""
    user_id: str
    click_history: List[Click]
    embedding: Optional[np.ndarray] = None


class PersonaConfig:
    """Configuration for persona generation."""
    def __init__(self, 
                 llm_api_key: str = "",
                 llm_model: str = "gemini-1.5-flash",
                 persona_model_path: str = "",
                 max_history_length: int = 50,
                 persona_dimensions: int = 128):
        self.llm_api_key = llm_api_key or os.getenv('GEMINI_API_KEY', '')
        self.llm_model = llm_model
        self.persona_model_path = persona_model_path
        self.max_history_length = max_history_length
        self.persona_dimensions = persona_dimensions


class PersonaEmbedder:
    """Generate user persona from click history."""
    
    def __init__(self, config: PersonaConfig):
        self.config = config
        self.persona_cache = {}
        self.last_persona_analysis = ""
    
    def generate_persona_from_history(self, clicked_articles: CandidateSet, non_clicked_articles: CandidateSet = None) -> np.ndarray:
        """Generate user persona from click history."""
        
        if not clicked_articles.articles:
            return np.zeros(self.config.persona_dimensions)
        
        # Take last N articles
        recent_articles = clicked_articles.articles[-self.config.max_history_length:]
        
        # Try LLM-based persona generation first
        if self.config.llm_api_key:
            try:
                return self._generate_persona_llm(recent_articles, non_clicked_articles)
            except Exception as e:
                logger.warning(f"LLM persona generation failed: {e}, falling back to simple method")
        
        # Fallback to simple aggregation
        return self._generate_persona_simple(recent_articles)
    
    def _generate_persona_llm(self, articles: List[Article], non_clicked_articles: CandidateSet = None) -> np.ndarray:
        """Generate comprehensive persona using Gemini LLM."""
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.config.llm_api_key)
            model = genai.GenerativeModel(self.config.llm_model)
            
            # Prepare article data
            clicked_titles = [article.title for article in articles]
            clicked_text = "\n".join([f"- {title}" for title in clicked_titles[-30:]])
            
            # Prepare non-clicked articles if available
            non_clicked_text = ""
            if non_clicked_articles and non_clicked_articles.articles:
                non_clicked_titles = [article.title for article in non_clicked_articles.articles]
                non_clicked_text = "\n".join([f"- {title}" for title in non_clicked_titles[:20]])
            
            # Enhanced comprehensive persona prompt with disengagement focus
            prompt = f"""
Generate a comprehensive user persona based on their news consumption patterns. Write this as a structured narrative with multiple focused paragraphs for each section.

CLICKED ARTICLES (User engaged with these):
{clicked_text}

{f'''
NON-CLICKED ARTICLES (User saw but ignored these):
{non_clicked_text}
''' if non_clicked_text else ''}

Create a detailed persona analysis in the following format:

**USER PERSONA SUMMARY**
Write 2-3 paragraphs describing the user's overall profile, primary characteristics, and what drives their news consumption behavior.

**PRIMARY INTERESTS & EXPERTISE LEVEL** 
Write 2-3 paragraphs detailing:
- Main topics they consistently engage with and why
- Their level of expertise (beginner, intermediate, expert) based on article complexity
- Specific subtopics and niches they prefer
- How their interests interconnect and influence each other

**READING BEHAVIOR & CONTENT PREFERENCES**
Write 2-3 paragraphs covering:
- Preferred content style (analytical vs narrative, breaking news vs deep analysis)
- Article length preferences and attention span indicators  
- Visual content engagement patterns
- Timing and frequency patterns in their reading habits

**DISENGAGEMENT PATTERNS & CONTENT AVOIDANCE**
Write 2-3 paragraphs focusing heavily on:
- Topics or content types this user consistently ignores or abandons
- Content characteristics that lead to quick exits (clickbait, certain writing styles, etc.)
- Competing priorities that cause disengagement from news
- Context patterns: when they disengage (time of day, after certain content, etc.)
- Emotional triggers that cause avoidance (controversial topics, negative news, etc.)

**ENGAGEMENT EVOLUTION & TRENDS**
Write 2-3 paragraphs analyzing:
- How their interests have evolved based on recent vs older clicks
- Seasonal or temporal patterns in their engagement
- Signs of growing or declining interest in specific areas
- Prediction of future interest development

**PERSONALIZATION STRATEGY**
Write 2-3 paragraphs recommending:
- Specific content types and topics to prioritize in recommendations
- Content formats and presentation styles that maximize engagement
- Topics and content characteristics to actively avoid
- Optimal timing and frequency for content delivery
- How to gradually introduce new topics based on their existing interests

Focus especially on disengagement analysis - understanding what they DON'T want is often more valuable than what they do want.
            """
            
            response = model.generate_content(prompt)
            persona_text = response.text
            
            # Store comprehensive persona analysis
            self.last_persona_analysis = persona_text
            logger.info(f"Generated comprehensive persona analysis ({len(persona_text)} chars)")
            
            # Convert text to vector representation
            return self._text_to_vector(persona_text)
            
        except Exception as e:
            logger.error(f"LLM persona generation failed: {e}")
            return self._generate_persona_simple(articles)
    
    def _generate_persona_simple(self, articles: List[Article]) -> np.ndarray:
        """Simple persona generation using article embeddings."""
        embeddings = []
        for article in articles:
            if article.embedding is not None:
                embeddings.append(article.embedding)
        
        if embeddings:
            # Apply recency weighting - more recent articles get higher weight
            weights = np.linspace(0.5, 1.0, len(embeddings))
            weighted_embeddings = np.array(embeddings) * weights.reshape(-1, 1)
            return np.mean(weighted_embeddings, axis=0).astype(np.float32)
        else:
            return np.zeros(self.config.persona_dimensions)
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector representation using hash-based method."""
        # Create multiple hash representations for better vector diversity
        hash_inputs = [
            text.encode(),
            (text + "_salt1").encode(),
            (text + "_salt2").encode(),
            (text + "_salt3").encode(),
        ]
        
        vectors = []
        for hash_input in hash_inputs:
            hash_obj = hashlib.md5(hash_input)
            hash_bytes = hash_obj.digest()
            vector_chunk = np.frombuffer(hash_bytes, dtype=np.uint8)
            vectors.extend(vector_chunk)
        
        # Convert to desired dimension
        vector = np.array(vectors[:self.config.persona_dimensions], dtype=np.float32)
        
        # Pad if necessary
        if len(vector) < self.config.persona_dimensions:
            vector = np.pad(vector, (0, self.config.persona_dimensions - len(vector)))
        
        # Normalize to [0, 1] range
        vector = vector / 255.0
        
        return vector


class PersonaScorer:
    """Score candidate articles based on user persona similarity."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha  # Weight for content similarity
        self.beta = beta    # Weight for engagement patterns
    
    def score_articles(self, 
                      candidate_articles: CandidateSet, 
                      user_persona: np.ndarray,
                      interest_profile: InterestProfile) -> CandidateSet:
        """Score candidate articles based on user persona."""
        
        if candidate_articles.embeddings is None:
            logger.warning("No article embeddings available for scoring")
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
        
        # Create scored result
        result = CandidateSet(
            articles=candidate_articles.articles,
            embeddings=candidate_articles.embeddings,
            scores=scores
        )
        
        return result
    
    def _compute_persona_scores(self, 
                              article_embeddings: np.ndarray, 
                              user_persona: np.ndarray,
                              interest_profile: InterestProfile) -> np.ndarray:
        """Compute similarity scores between user persona and articles."""
        
        # Ensure compatible dimensions
        if len(user_persona) != article_embeddings.shape[1]:
            if len(user_persona) > article_embeddings.shape[1]:
                user_persona = user_persona[:article_embeddings.shape[1]]
            else:
                user_persona = np.pad(
                    user_persona, 
                    (0, article_embeddings.shape[1] - len(user_persona))
                )
        
        # Normalize vectors for cosine similarity
        user_persona_norm = user_persona / (np.linalg.norm(user_persona) + 1e-8)
        article_norms = np.linalg.norm(article_embeddings, axis=1, keepdims=True)
        article_embeddings_norm = article_embeddings / (article_norms + 1e-8)
        
        # Compute content similarity scores
        content_scores = np.dot(article_embeddings_norm, user_persona_norm)
        
        # Compute engagement pattern scores
        engagement_scores = self._compute_engagement_scores(
            article_embeddings, interest_profile
        )
        
        # Combine scores
        final_scores = (self.alpha * content_scores + 
                       self.beta * engagement_scores)
        
        return np.clip(final_scores, 0.0, 1.0)
    
    def _compute_engagement_scores(self, 
                                 article_embeddings: np.ndarray,
                                 interest_profile: InterestProfile) -> np.ndarray:
        """Compute engagement-based scores."""
        
        num_articles = len(article_embeddings)
        engagement_scores = np.ones(num_articles) * 0.5
        
        if interest_profile.click_history:
            # Boost based on click history size (more clicks = more engagement)
            click_count = len(interest_profile.click_history)
            recency_boost = min(0.8 + (click_count / 50.0) * 0.2, 1.0)
            engagement_scores = engagement_scores * recency_boost
        
        return np.clip(engagement_scores, 0.0, 1.0)


def create_sample_data():
    """Create sample articles and user data for demonstration."""
    
    articles = [
        Article(
            "1", 
            "OpenAI Releases GPT-5: Revolutionary AI Model Transforms Natural Language Understanding",
            "The latest AI model from OpenAI demonstrates unprecedented capabilities in reasoning, creativity, and multimodal understanding."
        ),
        Article(
            "2",
            "Climate Summit 2024: World Leaders Agree on Ambitious Carbon Reduction Targets", 
            "Global climate conference results in binding commitments to reduce greenhouse gas emissions by 60% by 2030."
        ),
        Article(
            "3",
            "NFL Super Bowl 2024: Record-Breaking Viewership as Underdog Team Claims Victory",
            "The championship game attracts over 120 million viewers worldwide in thrilling overtime finish."
        ),
        Article(
            "4", 
            "Federal Reserve Cuts Interest Rates Amid Economic Uncertainty",
            "Central bank reduces rates by 0.5% to stimulate economic growth following mixed employment data."
        ),
        Article(
            "5",
            "Breakthrough Gene Therapy Cures Inherited Blindness in Clinical Trial",
            "Revolutionary treatment restores vision in 95% of patients with rare genetic disorder."
        ),
        Article(
            "6",
            "Tesla Unveils Fully Autonomous Robotaxi Fleet in Major Cities",
            "Electric vehicle manufacturer launches commercial self-driving taxi service in five metropolitan areas."
        ),
        Article(
            "7",
            "Antarctic Ice Sheet Melting Faster Than Previously Predicted, Study Warns",
            "Satellite data reveals accelerated ice loss could lead to more significant sea level rise by 2050."
        ),
        Article(
            "8", 
            "Apple Stock Soars Following Surprise AI Integration Announcement",
            "Tech giant's shares rise 12% after revealing comprehensive artificial intelligence strategy across product lineup."
        )
    ]
    
    # Generate random embeddings (in real use, these come from a trained model)
    embeddings = np.random.randn(len(articles), 128).astype(np.float32)
    for i, article in enumerate(articles):
        article.embedding = embeddings[i]
    
    return articles, embeddings


def main():
    """Run the persona-based recommender demo."""
    
    print("🤖 Persona-Based News Recommender Demo (Standalone)")
    print("=" * 60)
    
    # Load environment variables from .env file
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
        print("📋 Loaded environment variables from .env file")
    except FileNotFoundError:
        print("⚠️  No .env file found")
    except Exception as e:
        print(f"⚠️  Error loading .env file: {e}")
    
    # Create sample data
    articles, embeddings = create_sample_data()
    print(f"📰 Created {len(articles)} sample articles")
    
    # Simulate user click history (interested in AI/tech and climate)
    user_clicks = [
        Click("1"),  # OpenAI GPT-5
        Click("6"),  # Tesla robotaxi
        Click("8"),  # Apple AI integration
        Click("2"),  # Climate summit
        Click("7"),  # Antarctic melting
    ]
    
    clicked_articles = [articles[int(click.article_id)-1] for click in user_clicks]
    
    # Create non-clicked articles (articles user saw but ignored)
    clicked_ids = {click.article_id for click in user_clicks}
    non_clicked_articles = [article for article in articles if article.article_id not in clicked_ids]
    
    print(f"\n👤 User clicked on {len(user_clicks)} articles:")
    for click in user_clicks:
        article = articles[int(click.article_id)-1]
        print(f"   • {article.title[:60]}...")
    
    print(f"\n👁️  User saw but IGNORED {len(non_clicked_articles)} articles (disengagement data):")
    for article in non_clicked_articles[:3]:  # Show first 3
        print(f"   • {article.title[:60]}...")
    if len(non_clicked_articles) > 3:
        print(f"   • ... and {len(non_clicked_articles) - 3} more")
    
    # Configure and create persona embedder
    config = PersonaConfig(
        llm_api_key=os.getenv('GEMINI_API_KEY', ''),
        max_history_length=50,
        persona_dimensions=128
    )
    
    print(f"\n🔧 Configuration:")
    print(f"   • LLM API Key: {'✓ Available' if config.llm_api_key else '✗ Not found'}")
    if config.llm_api_key:
        print(f"     (Key: {config.llm_api_key[:10]}...{config.llm_api_key[-5:]})")
    print(f"   • Max history: {config.max_history_length}")
    print(f"   • Persona dimensions: {config.persona_dimensions}")
    
    # Debug: Show environment variable status
    env_key = os.getenv('GEMINI_API_KEY')
    print(f"\n🐛 Debug Info:")
    print(f"   • Environment GEMINI_API_KEY: {'✓ Found' if env_key else '✗ Not found'}")
    if env_key:
        print(f"     (Value: {env_key[:10]}...{env_key[-5:]})")
    
    # Generate persona
    embedder = PersonaEmbedder(config)
    clicked_set = CandidateSet(
        articles=clicked_articles,
        embeddings=np.array([a.embedding for a in clicked_articles])
    )
    
    non_clicked_set = CandidateSet(
        articles=non_clicked_articles,
        embeddings=np.array([a.embedding for a in non_clicked_articles])
    )
    
    print(f"\n🧠 Generating user persona (including disengagement analysis)...")
    try:
        persona = embedder.generate_persona_from_history(clicked_set, non_clicked_set)
        print(f"   ✓ Generated persona vector: {persona.shape}")
        print(f"   • Persona mean: {np.mean(persona):.4f}")
        print(f"   • Persona std: {np.std(persona):.4f}")
        
        # Show comprehensive analysis if available
        if embedder.last_persona_analysis:
            print(f"\n📊 Comprehensive Persona Analysis:")
            print("=" * 50)
            print(embedder.last_persona_analysis)
            print("=" * 50)
            
    except Exception as e:
        print(f"   ❌ Error generating persona: {e}")
        return
    
    # Score all candidate articles
    print(f"\n⚡ Scoring candidate articles...")
    
    scorer = PersonaScorer(alpha=0.7, beta=0.3)
    
    interest_profile = InterestProfile(
        user_id="demo_user",
        click_history=user_clicks
    )
    
    all_candidates = CandidateSet(
        articles=articles,
        embeddings=embeddings
    )
    
    try:
        scored_result = scorer.score_articles(all_candidates, persona, interest_profile)
        
        print(f"   ✓ Scored {len(scored_result.articles)} articles")
        
        # Sort by score and show recommendations
        if scored_result.scores is not None:
            sorted_indices = np.argsort(scored_result.scores)[::-1]
            
            print(f"\n🏆 Top 5 Personalized Recommendations:")
            for i, idx in enumerate(sorted_indices[:5]):
                article = scored_result.articles[idx]
                score = scored_result.scores[idx]
                clicked_marker = "👆" if article.article_id in [c.article_id for c in user_clicks] else "  "
                print(f"   {i+1}. [{score:.3f}] {clicked_marker} {article.title}")
            
            print(f"\n📈 Score Distribution:")
            scores = scored_result.scores
            print(f"   • High relevance (>0.6): {np.sum(scores > 0.6)} articles")
            print(f"   • Medium relevance (0.4-0.6): {np.sum((scores >= 0.4) & (scores <= 0.6))} articles")
            print(f"   • Low relevance (<0.4): {np.sum(scores < 0.4)} articles")
            print(f"   • Average score: {np.mean(scores):.3f}")
            
    except Exception as e:
        print(f"   ❌ Error scoring articles: {e}")
        return
    
    print(f"\n✨ Demo completed successfully!")
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   • Comprehensive persona generation from click history")
    print(f"   • LLM-based analysis with fallback methods")
    print(f"   • Persona-based article scoring and ranking")
    print(f"   • Engagement pattern consideration")
    print(f"   • Robust error handling")
    
    if not config.llm_api_key:
        print(f"\n💡 Tip: Set GEMINI_API_KEY environment variable to enable LLM-based persona analysis!")


if __name__ == "__main__":
    main()