#!/usr/bin/env python3
"""
Artifact comparison script that:
1. Loads downloaded artifacts from test_live_endpoint.py output
2. Makes LLM calls to compare specific pairs of artifacts
3. Persists comparison results with profile names and LLM outputs

Usage:
    python scripts/compare_artifacts.py [--input-dir DIR] [--output-file FILE] [--model MODEL]

Args:
    --input-dir: Directory containing profile artifacts (default: testing_data/evals/)
    --output-file: JSON file to save comparison results (default: testing_data/comparison_results.json)
    --model: LLM model to use for comparisons (default: gpt-4.1)
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import openai
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# response models
class ProfileUserModelComparison(BaseModel):
    user_model_diverges: bool
    explanation: str


class MissingArticle(BaseModel):
    headline: str
    explanation: str


class UserModelCandidatePoolComparison(BaseModel):
    missing_articles: list[MissingArticle]


class RankingComparison(BaseModel):
    misranked_articles: list[MissingArticle]


class AccuracyViolation(BaseModel):
    violation: str
    explanation: str


class AccuracyComparison(BaseModel):
    accuracy_violations: list[AccuracyViolation]


class SalienceComparison(BaseModel):
    salience_issues: list[str]
    explanation: str


class ArtifactComparator:
    def __init__(self, input_dir: str, output_file: str, model: str = "gpt-4.1"):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.model = model
        self.client = openai.OpenAI()

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

    def load_profile_artifacts(self, profile_dir: Path) -> Dict[str, Any]:
        """Load all artifacts for a single profile."""
        artifacts = {}
        profile_name = profile_dir.name

        # Load original profile JSON from tests/request_data/profiles/
        profile_json_file = Path("tests/request_data/profiles") / f"{profile_name}.json"
        if profile_json_file.exists():
            with open(profile_json_file, "r") as f:
                artifacts["original_profile"] = json.load(f)

        # Load user_model.txt
        user_model_file = profile_dir / "user_model.txt"
        if user_model_file.exists():
            with open(user_model_file, "r") as f:
                artifacts["user_model"] = f.read()

        # Load original_recommendations.pkl
        orig_rec_file = profile_dir / "original_recommendations.pkl"
        if orig_rec_file.exists():
            with open(orig_rec_file, "rb") as f:
                artifacts["original_recommendations"] = pickle.load(f)

        # Load rewritten_recommendations.pkl
        rewritten_rec_file = profile_dir / "rewritten_recommendations.pkl"
        if rewritten_rec_file.exists():
            with open(rewritten_rec_file, "rb") as f:
                artifacts["rewritten_recommendations"] = pickle.load(f)

        return artifacts

    def structure_interest_profile_from_original(self, original_profile: Dict[str, Any]) -> str:
        """Structure the interest profile using the exact same logic as _structure_interest_profile in the ranker."""
        click_history = original_profile.get("articles", [])
        if click_history:
            clicked_stories = sorted(
                [(art["headline"], art["published_at"]) for art in click_history], key=lambda x: x[1], reverse=True
            )
            clicked_headlines = [headline for headline, _ in clicked_stories]
        else:
            clicked_headlines = []

        interest_profile = original_profile.get("interest_profile", {})

        # Mirror the exact logic from _structure_interest_profile
        clean_profile = {
            "topics": [
                t.get("entity_name", "")
                for t in sorted(
                    interest_profile.get("onboarding_topics", []), key=lambda t: t.get("preference", 0), reverse=True
                )
                if t.get("entity_name")
            ],
            "click_topic_counts": interest_profile.get("click_topic_counts"),
            "click_locality_counts": interest_profile.get("click_locality_counts"),
            "click_history": clicked_headlines,
        }

        # Sort the click counts from most clicked to least clicked (exact logic from ranker)
        clean_profile["click_topic_counts"] = (
            [t for t, _ in sorted(clean_profile["click_topic_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_topic_counts"]
            else []
        )
        clean_profile["click_locality_counts"] = (
            [t for t, _ in sorted(clean_profile["click_locality_counts"].items(), key=lambda x: x[1], reverse=True)]
            if clean_profile["click_locality_counts"]
            else []
        )

        # Use the exact same string formatting as the ranker
        profile_str = f"""Topics the user has shown interest in (from most to least):
{", ".join(clean_profile["topics"])}

Topics the user has clicked on (from most to least):
{", ".join(clean_profile["click_topic_counts"])}

Localities the user has clicked on (from most to least):
{", ".join(clean_profile["click_locality_counts"])}

Headlines of articles the user has clicked on (most recent first):
{", ".join(cleaned_headline for cleaned_headline in clean_profile["click_history"] if cleaned_headline)}
"""

        return profile_str

    def compare_profile_to_user_model(self, profile: Dict, user_model: str) -> ProfileUserModelComparison:
        """Compare the structured interest profile to the user model."""
        profile_str = self.structure_interest_profile_from_original(profile)
        prompt = """
You are evaluating a colleague's description of a user based on a profile of their interactions and interests.
1. Compare the structured interest profile with the user model.
2. Identify any discrepancies or missing elements in the user model.
3. Provide a detailed explanation of how well the user model captures the user's interests.

Discrepancies should include:
- Missing topics or interests that are present in the profile
- Inaccurate representation of the user's click history
- Elevated or diminished importance of certain topics
- Any other notable differences
"""
        resp = self.client.responses.parse(
            model=self.model,
            instructions=prompt,
            input=f"Profile:\n{profile_str}\n\nUser Model:\n{user_model}",
            temperature=0.5,
            text_format=ProfileUserModelComparison,
        )

        return resp.output_parsed

    def compare_user_model_to_ranking(
        self, user_model: str, original_recs, profile
    ) -> tuple[UserModelCandidatePoolComparison, RankingComparison]:
        """Compare the user model to the ranking of articles."""
        candidate_pool = profile["candidates"]["articles"]
        candidate_pool_str = "\n".join([art["headline"] for art in candidate_pool])

        original_recs_headlines = [(rank + 1, i.headline) for rank, i in enumerate(original_recs.articles)]
        original_recs_str = "\n".join([f"{rank}. {headline}" for rank, headline in original_recs_headlines])

        prompt_pool = """
You are evaluating the outputs of a recommendation system.
Your task is to assess how well the selected articles align with the user's interests as described in the user model.

Keep in mind that the recommendation system is designed to balance relevance, diversity, and importance of articles, so not all selected articles will perfectly match the user model.

1. Compare the user model with the recommendations.
2. Identify any articles from the candidate pool that should have been recommended.
3. Provide a detailed explanation of the alignment between the user model and the recommendations.

"""

        prompt_faithfulness = """
You are evaluating the outputs of a recommendation system.
Your task is to assess how well the ranking of the selected articles aligns with the user's interests as described in the user model.
Keep in mind that the recommendation system is designed to balance relevance, diversity, and importance of articles, so not all selected articles will perfectly match the user model.

1. Compare the user model with the original recommendations.
2. Flag any articles that should have been ranked higher or lower based on the user model.
3. Provide a detailed explanation of the alignment between the user model and the recommendations.
"""

        response_pool = self.client.responses.parse(
            model=self.model,
            instructions=prompt_pool,
            input=f"User Model:\n{user_model}\n\nCandidate Pool:\n{candidate_pool_str}\n\nOriginal Recommendations:\n{original_recs_str}",
            temperature=0.5,
            text_format=UserModelCandidatePoolComparison,
        )

        response_faithfulness = self.client.responses.parse(
            model=self.model,
            instructions=prompt_faithfulness,
            input=f"User Model:\n{user_model}\n\nOriginal Recommendations:\n{original_recs_str}",
            temperature=0.5,
            text_format=RankingComparison,
        )

        return (
            response_pool.output_parsed,
            response_faithfulness.output_parsed,
        )

    def compare_user_model_to_rewritten_recommendations(
        self, profile: Dict, user_model: str, original_recs: Any, rewritten_recs: Any
    ) -> tuple[AccuracyComparison, SalienceComparison]:
        """Compare the user model to the rewritten recommendations."""
        article_ids = [str(i.article_id) for i in original_recs.articles]
        article_objects = []
        for i in article_ids:
            article_object = {}
            article_object["article_id"] = i
            article_object["headline_original"] = next(
                (art.headline for art in original_recs.articles if str(art.article_id) == i), ""
            )
            article_object["headline_rewritten"] = next(
                (art.headline for art in rewritten_recs.articles if str(art.article_id) == i), ""
            )
            article_object["body"] = next(
                (art["body"] for art in profile["candidates"]["articles"] if str(art["article_id"]) == i), ""
            )
            article_objects.append(article_object)

        accuracy_input = "\n\n".join(
            [f"Headline: {art['headline_rewritten']}\nArticle text: {art['body']}" for art in article_objects]
        )

        faithfulness_input = "\n\n".join(
            [
                f"Original Headline: {art['headline_original']}\nRewritten Headline: {art['headline_rewritten']}"
                for art in article_objects
            ]
        )
        faithfulness_input = f"User Model:\n{user_model}\n\n{faithfulness_input}"

        prompt_accuracy = """
You are an editor in a newsroom. Your job is to evaluate the accuracy of a headline based on the article text.

Look for inaccuracies, misleading statements, or any discrepancies between the headline and the article content.

1. For each rewritten headline, determine if it accurately reflects the content of the article.
2. Provide a detailed explanation of your assessment.
"""

        prompt_faithfulness = """
You are an audience engagement specialist. Your job is to evaluate how well the rewritten headlines align with the user's preferred tone and style as described in the user model.
Look for alignment, relevance, and how well the rewritten headlines capture the essence of the article.

1. For each rewritten headline, determine if it aligns with the user's interests as described in the user model.
2. Provide a detailed explanation of your assessment.
"""

        response_accuracy = self.client.responses.parse(
            model=self.model,
            instructions=prompt_accuracy,
            input=accuracy_input,
            temperature=0.5,
            text_format=AccuracyComparison,
        )

        response_faithfulness = self.client.responses.parse(
            model=self.model,
            instructions=prompt_faithfulness,
            input=faithfulness_input,
            temperature=0.5,
            text_format=SalienceComparison,
        )

        return (
            response_accuracy.output_parsed,
            response_faithfulness.output_parsed,
        )

    def process_profile(self, profile_dir: Path) -> Optional[Dict[str, Any]]:
        """Process a single profile and perform all comparisons."""
        profile_name = profile_dir.name
        logger.info(f"Processing profile: {profile_name}")

        # Load artifacts
        artifacts = self.load_profile_artifacts(profile_dir)

        if not artifacts:
            logger.warning(f"No artifacts found for profile {profile_name}")
            return None

        user_model_assessment = self.compare_profile_to_user_model(
            artifacts.get("original_profile", {}), artifacts.get("user_model", "")
        )

        ranking_assessment_pool, ranking_assessment_faithfulness = self.compare_user_model_to_ranking(
            artifacts.get("user_model", ""),
            artifacts.get("original_recommendations", None),
            artifacts.get("original_profile", {}),
        )

        rewrite_assessment_accuracy, rewrite_assessment_faithfulness = (
            self.compare_user_model_to_rewritten_recommendations(
                artifacts.get("original_profile", {}),
                artifacts.get("user_model", ""),
                artifacts.get("original_recommendations", None),
                artifacts.get("rewritten_recommendations", None),
            )
        )

        # Compile results
        results = {
            "profile_name": profile_name,
            "user_model": artifacts.get("user_model", ""),
            "ranked_headlines": "\n".join([art.headline for art in artifacts.get("original_recommendations").articles]),
            "rewritten_headlines": "\n".join(
                [art.headline for art in artifacts.get("rewritten_recommendations").articles]
            ),
            "user_model_assessment": user_model_assessment.model_dump(),
            "ranking_assessment_pool": ranking_assessment_pool.model_dump(),
            "ranking_assessment_faithfulness": ranking_assessment_faithfulness.model_dump(),
            "rewrite_assessment_accuracy": rewrite_assessment_accuracy.model_dump(),
            "rewrite_assessment_faithfulness": rewrite_assessment_faithfulness.model_dump(),
        }

        return results

    def run(self):
        """Run comparisons for all profiles."""
        # Find all profile directories
        profile_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(profile_dirs)} profile directories")

        all_results = []

        for profile_dir in profile_dirs:
            try:
                result = self.process_profile(profile_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {profile_dir}: {e}")
                continue

        # Save results
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "total_profiles": len(all_results),
            },
            "results": all_results,
        }

        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Comparison results saved to {self.output_file}")
        logger.info(f"Processed {len(all_results)} profiles successfully")


def main():
    parser = argparse.ArgumentParser(description="Compare artifacts and make LLM comparisons")
    parser.add_argument("--input-dir", default="testing_data/evals", help="Input directory with profile artifacts")
    parser.add_argument("--output-file", default="testing_data/comparison_results.json", help="Output file for results")
    parser.add_argument("--model", default="gpt-4.1", help="LLM model to use for comparisons")

    args = parser.parse_args()

    # Run the comparator
    comparator = ArtifactComparator(input_dir=args.input_dir, output_file=args.output_file, model=args.model)

    comparator.run()


if __name__ == "__main__":
    main()
