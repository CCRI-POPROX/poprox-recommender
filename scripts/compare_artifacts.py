#!/usr/bin/env python3
"""
Artifact comparison script that:
1. Loads downloaded artifacts from test_live_endpoint.py output
2. Makes LLM calls to compare specific pairs of artifacts
3. Persists comparison results with profile names and LLM outputs

Usage:
    python scripts/compare_artifacts.py [--input-dir DIR] [--output-file FILE] [--model MODEL]

Args:
    --input-dir: Directory containing profile artifacts (default: evaluation_data/)
    --output-file: JSON file to save comparison results (default: comparison_results.json)
    --model: LLM model to use for comparisons (default: gpt-4)
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import openai

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        interest_profile = original_profile.get("interest_profile", {})

        clicked_headlines = []  # We don't have headlines in the original profile data

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
            "click_history": clicked_headlines,  # Empty since we don't have headlines
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

    def make_llm_comparison(self, prompt: str, data1: str, data2: str) -> str:
        """Make an LLM call to compare two pieces of data."""
        full_prompt = f"""
{prompt}

Data 1:
{data1}

Data 2:
{data2}

Please provide your analysis:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing recommendation system data and user profiles.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or "No response content"
        except Exception as e:
            logger.error(f"Error making LLM call: {e}")
            return f"Error: {str(e)}"

    def compare_formatted_input_and_user_model(self, formatted_input: str, user_model: str) -> str:
        """Compare formatted input data and user model."""
        prompt = """
Please compare the formatted input profile data with the generated user model.
Analyze:
1. How well the user model captures the interests expressed in the input
2. Any gaps or discrepancies between input and model
3. The quality and completeness of the user model representation

PLACEHOLDER: Customize this prompt for your specific analysis needs.
"""
        return self.make_llm_comparison(prompt, formatted_input, user_model)

    def compare_user_model_and_original_recommendations(self, user_model: str, original_recs: Any) -> str:
        """Compare user model and original recommendations."""
        # Convert recommendations to string format for comparison
        recs_str = json.dumps(original_recs, indent=2, default=str) if original_recs else "No recommendations found"

        prompt = """
Please compare the user model with the original recommendations.
Analyze:
1. How well the recommendations align with the user model preferences
2. The relevance and quality of recommended articles
3. Any mismatches between user interests and recommendations

PLACEHOLDER: Customize this prompt for your specific analysis needs.
"""
        return self.make_llm_comparison(prompt, user_model, recs_str)

    def compare_user_model_and_rewritten_recommendations(self, user_model: str, rewritten_recs: Any) -> str:
        """Compare user model and rewritten recommendations (with updated headlines)."""
        # Convert recommendations to string format for comparison
        recs_str = (
            json.dumps(rewritten_recs, indent=2, default=str)
            if rewritten_recs
            else "No rewritten recommendations found"
        )

        prompt = """
Please compare the user model with the rewritten recommendations (with updated headlines).
Analyze:
1. How the headline rewrites affect recommendation relevance
2. Whether rewritten headlines better match user interests
3. The impact of headline changes on overall recommendation quality

PLACEHOLDER: Customize this prompt for your specific analysis needs.
"""
        return self.make_llm_comparison(prompt, user_model, recs_str)

    def process_profile(self, profile_dir: Path) -> Optional[Dict[str, Any]]:
        """Process a single profile and perform all comparisons."""
        profile_name = profile_dir.name
        logger.info(f"Processing profile: {profile_name}")

        # Load artifacts
        artifacts = self.load_profile_artifacts(profile_dir)

        if not artifacts:
            logger.warning(f"No artifacts found for profile {profile_name}")
            return None

        results = {"profile_name": profile_name, "timestamp": datetime.now().isoformat(), "comparisons": {}}

        # Extract formatted input data using the exact same logic as the ranker
        formatted_input = ""
        if "original_profile" in artifacts:
            formatted_input = self.structure_interest_profile_from_original(artifacts["original_profile"])

        # Comparison 1: Formatted input data vs User model
        if formatted_input and "user_model" in artifacts:
            logger.info(f"Comparing formatted input and user model for {profile_name}")
            results["comparisons"]["formatted_input_vs_user_model"] = self.compare_formatted_input_and_user_model(
                formatted_input, artifacts["user_model"]
            )

        # Comparison 2: User model vs Original recommendations
        if "user_model" in artifacts and "original_recommendations" in artifacts:
            logger.info(f"Comparing user model and original recommendations for {profile_name}")
            results["comparisons"]["user_model_vs_original_recs"] = (
                self.compare_user_model_and_original_recommendations(
                    artifacts["user_model"], artifacts["original_recommendations"]
                )
            )

        # Comparison 3: User model vs Rewritten recommendations
        if "user_model" in artifacts and "rewritten_recommendations" in artifacts:
            logger.info(f"Comparing user model and rewritten recommendations for {profile_name}")
            results["comparisons"]["user_model_vs_rewritten_recs"] = (
                self.compare_user_model_and_rewritten_recommendations(
                    artifacts["user_model"], artifacts["rewritten_recommendations"]
                )
            )

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
    parser.add_argument("--input-dir", default="evaluation_data", help="Input directory with profile artifacts")
    parser.add_argument("--output-file", default="comparison_results.json", help="Output file for results")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use for comparisons")

    args = parser.parse_args()

    # Run the comparator
    comparator = ArtifactComparator(input_dir=args.input_dir, output_file=args.output_file, model=args.model)

    comparator.run()


if __name__ == "__main__":
    main()
