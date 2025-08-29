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
                 max_history_length: int = 50,    # Keep original for compatibility
                 persona_dimensions: int = 128,
                 max_clicks_per_user: int = 50,       # Keep 50 clicks as requested
                 max_non_clicks_per_user: int = 200,  # Keep 200 non-clicks as requested
                 disengagement_threshold: int = 3,    # Keep original threshold
                 pattern_confidence_threshold: float = 0.7):  # Keep original confidence
        self.model_path = model_path
        self.device = device
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.persona_model_path = persona_model_path
        self.max_history_length = max_history_length
        self.persona_dimensions = persona_dimensions
        self.max_clicks_per_user = max_clicks_per_user
        self.max_non_clicks_per_user = max_non_clicks_per_user
        self.disengagement_threshold = disengagement_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold


class UserPersonaEmbedder(Component):
    """Generate user persona from click history and match with candidate articles."""

    config: UserPersonaConfig

    def __init__(self, config: UserPersonaConfig):
        super().__init__()
        self.config = config
        self.persona_cache = {}  # Cache generated personas
        self.last_persona_analysis = ""  # Store last comprehensive analysis

        # Load API key from environment if not provided
        if not self.config.llm_api_key:
            import os
            self.config.llm_api_key = os.getenv('GEMINI_API_KEY', '')

    def generate_persona_from_history(self, clicked_articles: CandidateSet, non_clicked_articles: CandidateSet = None) -> np.ndarray:
        """Generate user persona from both clicked and non-clicked articles."""

        if len(clicked_articles.articles) == 0:
            return np.zeros(self.config.persona_dimensions)

        # Take recent clicked articles (up to 50)
        recent_clicked = clicked_articles.articles[-self.config.max_clicks_per_user:]

        # Take recent non-clicked articles for disengagement analysis (up to 200)
        recent_non_clicked = []
        if non_clicked_articles and len(non_clicked_articles.articles) > 0:
            recent_non_clicked = non_clicked_articles.articles[-self.config.max_non_clicks_per_user:]

        # Debug tracking
        self._debug_clicked_count = len(recent_clicked)
        self._debug_non_clicked_count = len(recent_non_clicked)
        if len(recent_clicked) > 0:
            self._debug_final_ratio = f"1:{len(recent_non_clicked)/len(recent_clicked):.1f}"
        else:
            self._debug_final_ratio = "N/A"

        logger.info(f"Persona generation: processing {len(recent_clicked)} clicked + {len(recent_non_clicked)} non-clicked articles")
        logger.info(f"Limits applied: max_clicks={self.config.max_clicks_per_user}, max_non_clicks={self.config.max_non_clicks_per_user}")
        logger.info(f"Truncation: clicked {len(clicked_articles.articles)}→{len(recent_clicked)}, non-clicked {len(non_clicked_articles.articles) if non_clicked_articles else 0}→{len(recent_non_clicked)}")

        # Option 1: LLM-based persona generation with disengagement analysis
        if self.config.llm_api_key:
            return self._generate_persona_llm(recent_clicked, recent_non_clicked)

        # Option 2: Pre-trained model persona generation
        elif self.config.persona_model_path:
            return self._generate_persona_pretrained(recent_clicked, recent_non_clicked)

        # Fallback: Simple aggregation
        else:
            return self._generate_persona_simple(recent_clicked, recent_non_clicked)

    def _generate_persona_llm(self, clicked_articles: list, non_clicked_articles: list = []) -> np.ndarray:
        """Generate persona using Gemini LLM with comprehensive disengagement analysis."""
        try:
            import google.generativeai as genai

            # Configure Gemini
            genai.configure(api_key=self.config.llm_api_key)
            model = genai.GenerativeModel(self.config.llm_model)

            # Process clicked articles (keep only recent 15 for efficiency)
            clicked_titles = [article.title for article in clicked_articles if hasattr(article, 'title')]
            clicked_text = "\n".join([f"- {title}" for title in clicked_titles[-15:]])

            # Process non-clicked articles (keep only recent 15)
            non_clicked_titles = [article.title for article in non_clicked_articles if hasattr(article, 'title')]
            non_clicked_text = "\n".join([f"- {title}" for title in non_clicked_titles[-15:]])

            # COMPACT persona prompt optimized for daily newsletters
            prompt = f"""
Analyze this user's news reading pattern for daily newsletter personalization:

CLICKED ARTICLES (User's interests):
{clicked_text}

IGNORED ARTICLES (User avoided these):
{non_clicked_text if non_clicked_text else "Limited data"}


Create a user persona in 3 short paragraphs, focusing on detailed content preferences:
1. INTERESTS: What specific content, finer-grained entities, and granular sub-categories does this user prefer? (1–5 sentences)
2. AVOIDS: What specific content and finer-grained entities does this user consistently skip? (1–5 sentences)
3. DIVERSITY: What are the user’s content and entities diversity preferences? (1–2 sentences)
4. RECOMMENDATIONS: For daily newsletters, prioritize X content, avoid Y content, and take into account the user’s content and entities diversity preferences. (1–5 sentences)

Keep the total response under 500 words.
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
            return self._generate_persona_simple(clicked_articles, non_clicked_articles)

    def _generate_persona_pretrained(self, clicked_articles: list, non_clicked_articles: list = []) -> np.ndarray:
        """Generate persona using pre-trained model."""
        try:
            from sentence_transformers import SentenceTransformer

            # Load pre-trained sentence transformer
            model = SentenceTransformer(self.config.persona_model_path or 'all-MiniLM-L6-v2')

            # Extract article titles and content from clicked articles
            clicked_texts = []
            for article in clicked_articles:
                if hasattr(article, 'title') and hasattr(article, 'abstract'):
                    text = f"LIKED: {article.title}. {article.abstract}"
                elif hasattr(article, 'title'):
                    text = f"LIKED: {article.title}"
                else:
                    continue
                clicked_texts.append(text)

            # Extract non-clicked articles for disengagement patterns
            non_clicked_texts = []
            for article in non_clicked_articles:
                if hasattr(article, 'title') and hasattr(article, 'abstract'):
                    text = f"IGNORED: {article.title}. {article.abstract}"
                elif hasattr(article, 'title'):
                    text = f"IGNORED: {article.title}"
                else:
                    continue
                non_clicked_texts.append(text)

            # Combine all article texts for comprehensive analysis
            article_texts = clicked_texts + non_clicked_texts

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
            return self._generate_persona_simple(clicked_articles, non_clicked_articles)
        except Exception as e:
            logger.error(f"Pre-trained persona generation failed: {e}")
            return self._generate_persona_simple(clicked_articles, non_clicked_articles)

    def _generate_persona_simple(self, clicked_articles: list, non_clicked_articles: list = []) -> np.ndarray:
        """Simple persona generation using article embeddings with disengagement weighting."""
        # Aggregate clicked article embeddings (positive signal)
        clicked_embeddings = []
        for article in clicked_articles:
            if hasattr(article, 'embedding') and article.embedding is not None:
                clicked_embeddings.append(article.embedding)

        # Aggregate non-clicked article embeddings (negative signal)
        non_clicked_embeddings = []
        for article in non_clicked_articles:
            if hasattr(article, 'embedding') and article.embedding is not None:
                non_clicked_embeddings.append(article.embedding)

        # Create persona by emphasizing clicked content and reducing non-clicked patterns
        if clicked_embeddings:
            liked_persona = np.mean(clicked_embeddings, axis=0)

            if non_clicked_embeddings:
                # Reduce influence of disengaged content patterns
                disliked_persona = np.mean(non_clicked_embeddings, axis=0)
                # Persona = what they like - 0.3 * what they ignore
                persona = liked_persona - 0.3 * disliked_persona
                return persona
            else:
                return liked_persona
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
            interest_profile: InterestProfile, historical_newsletters: CandidateSet = None, **kwargs) -> np.ndarray:
        """Generate user persona from historical newsletter engagement patterns."""

        # Extract disengagement patterns from historical newsletters
        if historical_newsletters is not None:
            non_clicked_articles = self._extract_disengagement_patterns(
                historical_newsletters, clicked_articles, interest_profile
            )
        else:
            # Fallback: create empty non-clicked set
            logger.warning("No historical newsletter data provided - using clicked-only persona")
            non_clicked_articles = self._create_empty_candidate_set()

        # Generate persona from both engagement and disengagement patterns
        persona = self.generate_persona_from_history(clicked_articles, non_clicked_articles)

        # Cache the persona
        user_id = getattr(interest_profile, 'user_id', 'default')
        self.persona_cache[user_id] = persona

        return persona

    def _extract_disengagement_patterns(self, historical_newsletters: CandidateSet,
                                        clicked_articles: CandidateSet,
                                        interest_profile: InterestProfile) -> CandidateSet:
        """Extract non-clicked articles from historical newsletters where user actually engaged."""

        # STEP 1: Get ALL clicked article IDs from user's complete history
        all_clicked_ids = {article.article_id for article in clicked_articles.articles
                          if hasattr(article, 'article_id')}

        logger.info(f"User has {len(all_clicked_ids)} total clicks in history")

        # STEP 2: Group historical newsletter articles by newsletter batch
        newsletter_groups = self._group_articles_by_newsletter(historical_newsletters)
        logger.info(f"Found {len(newsletter_groups)} historical newsletter batches")

        # STEP 3: Find newsletters where user had ANY engagement (≥1 click)
        engaged_newsletters = []
        total_engaged_articles = 0
        total_non_clicked = 0

        for newsletter_id, newsletter_articles in newsletter_groups.items():
            # Count how many articles from this newsletter the user clicked
            clicked_in_newsletter = [a for a in newsletter_articles
                                   if getattr(a, 'article_id', None) in all_clicked_ids]

            if len(clicked_in_newsletter) >= 1:  # User engaged with this newsletter
                # Extract articles from this newsletter that user DIDN'T click
                non_clicked_in_newsletter = [a for a in newsletter_articles
                                           if getattr(a, 'article_id', None) not in all_clicked_ids]

                engaged_newsletters.append({
                    'newsletter_id': newsletter_id,
                    'total_articles': len(newsletter_articles),
                    'clicked_articles': clicked_in_newsletter,
                    'non_clicked_articles': non_clicked_in_newsletter,
                    'engagement_rate': len(clicked_in_newsletter) / len(newsletter_articles)
                })

                total_engaged_articles += len(newsletter_articles)
                total_non_clicked += len(non_clicked_in_newsletter)

        logger.info(f"Found {len(engaged_newsletters)} engaged newsletters with {total_non_clicked} non-clicked articles")

        # STEP 4: Sample non-clicked articles intelligently (up to 200)
        sampled_non_clicked = self._sample_non_clicked_articles_corrected(engaged_newsletters)

        # STEP 5: Apply statistical pattern filtering
        recent_clicked_articles = clicked_articles.articles[-self.config.max_history_length:]
        filtered_non_clicked = self._filter_disengagement_patterns_robust(sampled_non_clicked, recent_clicked_articles)

        # STEP 6: Get embeddings for filtered articles
        filtered_embeddings = self._get_embeddings_for_articles(filtered_non_clicked, historical_newsletters)

        logger.info(f"Final result: {len(filtered_non_clicked)} high-confidence disengagement articles")

        return self._create_candidate_set(filtered_non_clicked, filtered_embeddings)

    def _sample_non_clicked_articles(self, relevant_newsletters: list) -> list:
        """Intelligently sample non-clicked articles to maintain balance."""
        import random

        all_non_clicked = []

        # Collect all non-clicked articles with metadata
        for newsletter_id, articles, click_count in relevant_newsletters:
            for article in articles:
                all_non_clicked.append((article, newsletter_id, click_count))

        # If within limit, return all
        if len(all_non_clicked) <= self.config.max_non_clicks_per_user:
            return [item[0] for item in all_non_clicked]

        # Intelligent sampling strategy:
        # 1. Prioritize articles from newsletters with more engagement (higher click_count)
        # 2. Ensure topic diversity
        # 3. Recent articles get priority

        # Sort by newsletter engagement level (descending)
        all_non_clicked.sort(key=lambda x: x[2], reverse=True)

        # Sample with stratification by topic
        sampled = []
        topic_counts = {}
        max_per_topic = max(5, self.config.max_non_clicks_per_user // 10)  # Max 5 articles per topic

        for article, newsletter_id, click_count in all_non_clicked:
            if len(sampled) >= self.config.max_non_clicks_per_user:
                break

            article_topics = self._get_article_topics(article)
            primary_topic = article_topics[0] if article_topics else 'unknown'

            # Limit per topic to ensure diversity
            if topic_counts.get(primary_topic, 0) < max_per_topic:
                sampled.append(article)
                topic_counts[primary_topic] = topic_counts.get(primary_topic, 0) + 1

        # If still under limit, add remaining randomly
        remaining_articles = [item[0] for item in all_non_clicked if item[0] not in sampled]
        if len(sampled) < self.config.max_non_clicks_per_user and remaining_articles:
            additional_needed = self.config.max_non_clicks_per_user - len(sampled)
            random.shuffle(remaining_articles)
            sampled.extend(remaining_articles[:additional_needed])

        logger.info(f"Sampled {len(sampled)} non-clicked articles from {len(all_non_clicked)} total")
        return sampled

    def _sample_non_clicked_articles_corrected(self, engaged_newsletters: list) -> list:
        """Corrected intelligent sampling of non-clicked articles with 1:4 ratio (50:200)."""
        import random

        # Collect all non-clicked articles with metadata
        all_non_clicked = []
        for newsletter_data in engaged_newsletters:
            newsletter_id = newsletter_data['newsletter_id']
            engagement_rate = newsletter_data['engagement_rate']
            non_clicked_articles = newsletter_data['non_clicked_articles']

            for article in non_clicked_articles:
                all_non_clicked.append({
                    'article': article,
                    'newsletter_id': newsletter_id,
                    'engagement_rate': engagement_rate,  # Higher = more reliable signal
                    'newsletter_size': len(newsletter_data['non_clicked_articles'])
                })

        logger.info(f"Total non-clicked articles available: {len(all_non_clicked)}")

        # If within limit, return all
        if len(all_non_clicked) <= self.config.max_non_clicks_per_user:
            return [item['article'] for item in all_non_clicked]

        # SAMPLING STRATEGY for 200 articles:
        # 1. Priority to newsletters with higher engagement rates (stronger signal)
        # 2. Ensure topic diversity (max 20 per topic)
        # 3. Recency preference (if timestamp available)
        # 4. Balance across newsletters

        # Sort by engagement rate (descending) for priority
        all_non_clicked.sort(key=lambda x: x['engagement_rate'], reverse=True)

        sampled = []
        topic_counts = {}
        newsletter_counts = {}
        max_per_topic = 20  # Max 20 articles per topic (200/10 topics)
        max_per_newsletter = 50  # Max 50 per newsletter to ensure diversity

        # Sample with constraints
        for item in all_non_clicked:
            if len(sampled) >= self.config.max_non_clicks_per_user:
                break

            article = item['article']
            newsletter_id = item['newsletter_id']

            # Get article topics
            article_topics = self._get_article_topics(article)
            primary_topic = article_topics[0] if article_topics else 'unknown'

            # Apply constraints
            topic_count = topic_counts.get(primary_topic, 0)
            newsletter_count = newsletter_counts.get(newsletter_id, 0)

            if (topic_count < max_per_topic and
                newsletter_count < max_per_newsletter):

                sampled.append(article)
                topic_counts[primary_topic] = topic_count + 1
                newsletter_counts[newsletter_id] = newsletter_count + 1

        # If still under target, fill remaining with random selection
        if len(sampled) < self.config.max_non_clicks_per_user:
            remaining_items = [item for item in all_non_clicked
                             if item['article'] not in sampled]
            remaining_needed = self.config.max_non_clicks_per_user - len(sampled)

            if remaining_items:
                random.shuffle(remaining_items)
                additional_articles = [item['article'] for item in remaining_items[:remaining_needed]]
                sampled.extend(additional_articles)

        logger.info(f"Sampled {len(sampled)}/{self.config.max_non_clicks_per_user} non-clicked articles")
        logger.info(f"Topic distribution: {dict(list(topic_counts.items())[:5])}")  # Show top 5 topics

        return sampled

    def _get_embeddings_for_articles(self, articles: list, historical_newsletters: CandidateSet) -> list:
        """Get embeddings for filtered articles."""
        embeddings = []

        for article in articles:
            article_id = getattr(article, 'article_id', None)
            if (hasattr(historical_newsletters, 'embeddings') and
                historical_newsletters.embeddings is not None and article_id):

                # Find article index in historical_newsletters
                article_index = next(
                    (idx for idx, hist_art in enumerate(historical_newsletters.articles)
                     if getattr(hist_art, 'article_id', None) == article_id),
                    None
                )

                if article_index is not None and article_index < len(historical_newsletters.embeddings):
                    embeddings.append(historical_newsletters.embeddings[article_index])

        return embeddings

    def _group_articles_by_newsletter(self, historical_newsletters: CandidateSet) -> dict:
        """Group historical articles by newsletter/batch."""
        # This assumes articles have newsletter_id, timestamp, or batch_id
        # If not available, we'll use date-based grouping

        groups = {}
        for article in historical_newsletters.articles:
            # Try multiple ways to identify newsletter groups
            newsletter_id = (getattr(article, 'newsletter_id', None) or
                           getattr(article, 'batch_id', None) or
                           getattr(article, 'published_date', 'unknown'))

            if newsletter_id not in groups:
                groups[newsletter_id] = []
            groups[newsletter_id].append(article)

        return groups

    def _filter_disengagement_patterns_robust(self, non_clicked_articles: list, recent_clicked_articles: list) -> list:
        """Apply robust statistical filtering to find high-confidence disengagement patterns."""

        if len(non_clicked_articles) < self.config.disengagement_threshold:
            logger.warning(f"Insufficient non-clicked articles ({len(non_clicked_articles)}) for pattern analysis")
            return non_clicked_articles

        # Count topic frequencies with statistical significance
        clicked_topics = self._extract_topics(recent_clicked_articles)  # Use recent clicks only
        non_clicked_topics = self._extract_topics(non_clicked_articles)

        total_clicked = len(recent_clicked_articles)
        total_non_clicked = len(non_clicked_articles)

        # Statistical analysis of disengagement patterns
        avoided_topics = set()
        strongly_preferred_topics = set()

        for topic, non_click_count in non_clicked_topics.items():
            click_count = clicked_topics.get(topic, 0)

            # Calculate proportions
            click_proportion = click_count / total_clicked if total_clicked > 0 else 0
            non_click_proportion = non_click_count / total_non_clicked

            # Statistical significance thresholds
            min_occurrences = max(3, int(total_non_clicked * 0.02))  # At least 2% of non-clicks

            # Strong avoidance pattern: high in non-clicks, low in clicks, statistically significant
            if (non_click_count >= min_occurrences and
                click_proportion < 0.1 and  # Less than 10% of clicks
                non_click_proportion > 0.05 and  # At least 5% of non-clicks
                non_click_count / (click_count + 1) > 3):  # Strong ratio (avoid division by 0)

                # Confidence score based on frequency and ratio
                confidence = min(1.0, (non_click_count / min_occurrences) * (non_click_proportion / 0.05))
                if confidence >= self.config.pattern_confidence_threshold:
                    avoided_topics.add(topic)

        # Also identify strongly preferred topics for contrast
        for topic, click_count in clicked_topics.items():
            non_click_count = non_clicked_topics.get(topic, 0)

            if (click_count >= 2 and  # At least 2 clicks
                click_count / total_clicked > 0.08):  # At least 8% of clicks
                strongly_preferred_topics.add(topic)

        logger.info(f"High-confidence avoided topics ({len(avoided_topics)}): {avoided_topics}")
        logger.info(f"Strongly preferred topics ({len(strongly_preferred_topics)}): {strongly_preferred_topics}")

        # Filter articles: prioritize avoided topics, exclude preferred topics from non-clicked
        filtered_articles = []
        for article in non_clicked_articles:
            article_topics = self._get_article_topics(article)

            # Include if article contains avoided topics
            if any(topic in avoided_topics for topic in article_topics):
                filtered_articles.append(article)
            # Exclude if article contains strongly preferred topics (might be misclassified)
            elif not any(topic in strongly_preferred_topics for topic in article_topics):
                # Include some general articles for broader context, but with lower priority
                if len(filtered_articles) < self.config.max_non_clicks_per_user * 0.8:  # 80% limit
                    filtered_articles.append(article)

        # Ensure we have enough variety - add some random samples if too restrictive
        if len(filtered_articles) < min(20, len(non_clicked_articles) // 3):
            remaining = [a for a in non_clicked_articles if a not in filtered_articles]
            import random
            random.shuffle(remaining)
            needed = min(20, len(non_clicked_articles) // 3) - len(filtered_articles)
            filtered_articles.extend(remaining[:needed])

        logger.info(f"Filtered to {len(filtered_articles)} high-confidence disengagement articles")
        return filtered_articles

    def _extract_topics(self, articles: list) -> dict:
        """Extract topics from articles and count frequencies."""
        topic_counts = {}
        for article in articles:
            topics = self._get_article_topics(article)
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return topic_counts

    def _get_article_topics(self, article) -> list:
        """Extract topics from an article (placeholder - adapt to your article structure)."""
        # This should be adapted based on how topics are stored in your articles
        # Common approaches: category field, topic classification, keyword extraction

        topics = []

        # Method 1: Direct category field
        if hasattr(article, 'category'):
            topics.append(article.category.lower())

        # Method 2: Topic classification from title
        if hasattr(article, 'title'):
            title_lower = article.title.lower()
            if any(word in title_lower for word in ['sport', 'game', 'championship', 'team']):
                topics.append('sports')
            elif any(word in title_lower for word in ['celebrity', 'fashion', 'entertainment']):
                topics.append('entertainment')
            elif any(word in title_lower for word in ['tech', 'ai', 'technology', 'computer']):
                topics.append('technology')
            elif any(word in title_lower for word in ['climate', 'environment', 'green', 'carbon']):
                topics.append('environment')
            elif any(word in title_lower for word in ['politics', 'government', 'policy', 'election']):
                topics.append('politics')
            else:
                topics.append('general')

        return topics if topics else ['unknown']

    def _create_empty_candidate_set(self):
        """Create an empty CandidateSet."""
        try:
            return CandidateSet(articles=[])
        except TypeError:
            # Fallback for testing
            class MockCandidateSet:
                def __init__(self, articles):
                    self.articles = articles
                    self.embeddings = None
            return MockCandidateSet(articles=[])

    def _create_candidate_set(self, articles: list, embeddings: list = None):
        """Create CandidateSet with proper error handling."""
        try:
            candidate_set = CandidateSet(articles=articles)
            if embeddings:
                import numpy as np
                candidate_set.embeddings = np.array(embeddings)
            return candidate_set
        except TypeError:
            # Fallback for testing
            class MockCandidateSet:
                def __init__(self, articles):
                    self.articles = articles
                    self.embeddings = None
            mock_set = MockCandidateSet(articles=articles)
            if embeddings:
                import numpy as np
                mock_set.embeddings = np.array(embeddings)
            return mock_set
