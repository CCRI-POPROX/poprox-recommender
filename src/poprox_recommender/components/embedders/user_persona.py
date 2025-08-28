# pyright: basic

import logging
import numpy as np

try:
    from lenskit.pipeline import Component
    from poprox_concepts import CandidateSet, InterestProfile
    from poprox_recommender.components.embedders.user import NRMSUserEmbedder, NRMSUserEmbedderConfig
except ImportError:
    # Fallback for testing without full dependencies
    Component = object
    CandidateSet = object
    InterestProfile = object
    NRMSUserEmbedder = object
    NRMSUserEmbedderConfig = object

logger = logging.getLogger(__name__)


class UserPersonaConfig:
    """Configuration for user persona generation."""

    def __init__(self,
                 model_path: str,
                 device: str = "cpu",
                 llm_api_key: str = "",
                 llm_model: str = "gemini-1.5-flash",
                 persona_model_path: str = "",
                 max_history_length: int = 50,
                 persona_dimensions: int = 128,
                 max_clicks_per_user: int = 50):
        self.model_path = model_path
        self.device = device
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.persona_model_path = persona_model_path
        self.max_history_length = max_history_length
        self.persona_dimensions = persona_dimensions
        self.max_clicks_per_user = max_clicks_per_user


class UserPersonaEmbedder(Component):
    """Generate user persona from click history and match with candidate articles."""

    def __init__(self, config: UserPersonaConfig):
        super().__init__()
        self.config = config
        self.persona_cache = {}  # Cache generated personas
        self.last_persona_analysis = ""  # Store last comprehensive analysis
        
        # Load API key from environment if not provided
        if not self.config.llm_api_key:
            import os
            self.config.llm_api_key = os.getenv('GEMINI_API_KEY', '')

    def generate_persona_from_history(self, clicked_articles: CandidateSet) -> np.ndarray:
        """Generate user persona from click history."""

        if len(clicked_articles.articles) == 0:
            return np.zeros(self.config.persona_dimensions)

        # Take last 50 articles
        recent_articles = clicked_articles.articles[-self.config.max_history_length:]

        # Option 1: LLM-based persona generation
        if self.config.llm_api_key:
            return self._generate_persona_llm(recent_articles)

        # Option 2: Pre-trained model persona generation
        elif self.config.persona_model_path:
            return self._generate_persona_pretrained(recent_articles)

        # Fallback: Simple aggregation
        else:
            return self._generate_persona_simple(recent_articles)

    def _generate_persona_llm(self, articles: list) -> np.ndarray:
        """Generate persona using Gemini LLM with comprehensive analysis."""
        try:
            import google.generativeai as genai

            # Configure Gemini
            genai.configure(api_key=self.config.llm_api_key)
            model = genai.GenerativeModel(self.config.llm_model)

            # Separate clicked and non-clicked articles for comprehensive analysis
            clicked_titles = [article.title for article in articles if hasattr(article, 'title')]
            clicked_text = "\n".join([f"- {title}" for title in clicked_titles[-30:]])

            # Enhanced comprehensive persona prompt with disengagement focus
            prompt = f"""
Generate a comprehensive user persona based on their news consumption patterns. Write this as a structured narrative with multiple focused paragraphs for each section.

CLICKED ARTICLES (User engaged with these):
{clicked_text}

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

            # Store comprehensive persona text for analysis
            self.last_persona_analysis = persona_text
            logger.info(f"Generated comprehensive persona: {persona_text[:200]}...")

            # Convert text to vector representation
            return self._text_to_vector(persona_text)

        except Exception as e:
            logger.error(f"LLM persona generation failed: {e}")
            return self._generate_persona_simple(articles)

    def _generate_persona_pretrained(self, articles: list) -> np.ndarray:
        """Generate persona using pre-trained model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load pre-trained sentence transformer
            model = SentenceTransformer(self.config.persona_model_path or 'all-MiniLM-L6-v2')
            
            # Extract article titles and content
            article_texts = []
            for article in articles:
                if hasattr(article, 'title') and hasattr(article, 'abstract'):
                    text = f"{article.title}. {article.abstract}"
                elif hasattr(article, 'title'):
                    text = article.title
                else:
                    continue
                article_texts.append(text)
            
            if not article_texts:
                return np.zeros(self.config.persona_dimensions)
            
            # Generate embeddings for all articles
            embeddings = model.encode(article_texts)
            
            # Create user persona by averaging article embeddings
            # Apply recency weighting - more recent articles get higher weight
            weights = np.linspace(0.5, 1.0, len(embeddings))
            weighted_embeddings = embeddings * weights.reshape(-1, 1)
            persona_embedding = np.mean(weighted_embeddings, axis=0)
            
            # Normalize and resize to match desired dimensions
            persona_embedding = persona_embedding / (np.linalg.norm(persona_embedding) + 1e-8)
            
            # Resize to match desired persona dimensions
            if len(persona_embedding) > self.config.persona_dimensions:
                persona_embedding = persona_embedding[:self.config.persona_dimensions]
            elif len(persona_embedding) < self.config.persona_dimensions:
                persona_embedding = np.pad(
                    persona_embedding,
                    (0, self.config.persona_dimensions - len(persona_embedding))
                )
            
            return persona_embedding.astype(np.float32)
            
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to simple method")
            return self._generate_persona_simple(articles)
        except Exception as e:
            logger.error(f"Pre-trained persona generation failed: {e}")
            return self._generate_persona_simple(articles)

    def _generate_persona_simple(self, articles: list) -> np.ndarray:
        """Simple persona generation using article embeddings."""
        # Aggregate article embeddings to create user persona
        embeddings = []
        for article in articles:
            if hasattr(article, 'embedding') and article.embedding is not None:
                embeddings.append(article.embedding)

        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.config.persona_dimensions)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector representation."""
        # Simple hash-based approach - you can enhance this
        import hashlib

        # Create a deterministic vector from text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to numpy array
        vector = np.frombuffer(hash_bytes, dtype=np.uint8)

        # Pad or truncate to desired dimension
        if len(vector) < self.config.persona_dimensions:
            vector = np.pad(vector, (0, self.config.persona_dimensions - len(vector)))
        else:
            vector = vector[:self.config.persona_dimensions]

        # Normalize
        return vector.astype(np.float32) / 255.0

    def run(self, candidate_articles: CandidateSet, clicked_articles: CandidateSet,
            interest_profile: InterestProfile, **kwargs) -> np.ndarray:
        """Generate user persona and return it."""

        # Generate persona from click history
        persona = self.generate_persona_from_history(clicked_articles)

        # Cache the persona
        user_id = getattr(interest_profile, 'user_id', 'default')
        self.persona_cache[user_id] = persona

        return persona
