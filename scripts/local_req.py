# Simulates a request to the recommender without requiring Serverless
import json
import time
import uuid
import warnings

from poprox_concepts.api.recommendations.v2 import RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.main import root
from poprox_recommender.paths import project_root
from poprox_recommender.recommenders.load import discover_pipelines

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    project_root_path = project_root()
    profile_paths = (project_root_path / "tests" / "request_data" / "profiles").glob("*.json")

    # Get all available pipelines
    available_pipelines = discover_pipelines()
    print(f"Available pipelines: {available_pipelines}")

    all_results = []

    for profile_path in profile_paths:
        print(f"Loading profile from {profile_path}")

        with open(profile_path, "r") as profile_file:
            raw_json = profile_file.read()
            req = RecommendationRequestV2.model_validate_json(raw_json)

        # Run each pipeline for each profile
        for pipeline_name in available_pipelines:
            print(f"  Testing pipeline: {pipeline_name}")

            # Run each pipeline 3 times for more accurate timing data
            for run_num in range(1, 4):
                print(f"    Run {run_num}/3")

                # Create unique request with new UUID to avoid overwrites
                req_copy = req.model_copy()
                req_copy.interest_profile.profile_id = uuid.uuid4()

                print(f"      Profile ID: {req_copy.interest_profile.profile_id}")
                print(f"      Number of candidates: {len(req_copy.candidates.articles)}")
                print(f"      Requested recommendations: {req_copy.num_recs}")

                # Time the request
                start_time = time.time()
                try:
                    response = root(req_copy.model_dump(), pipeline=pipeline_name)
                    end_time = time.time()
                    execution_time = end_time - start_time

                    print(f"      Raw response type: {type(response)}")
                    print(
                        f"      Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}"
                    )

                    # Handle the response - it's already a dict from root()
                    if isinstance(response, dict):
                        response = RecommendationResponseV2.model_validate(response)
                    else:
                        # If it's not a dict, convert it
                        response = RecommendationResponseV2.model_validate(
                            response.model_dump() if hasattr(response, "model_dump") else response
                        )

                    print(f"      Execution time: {execution_time:.3f}s")
                    print(f"      Recommendations returned: {len(response.recommendations.articles)}")

                except Exception as e:
                    print(f"      ERROR: {type(e).__name__}: {e}")
                    print(f"      Pipeline: {pipeline_name}")
                    print(f"      Profile: {profile_path.stem}")
                    raise

                # Map article_id to original headline for quick lookup
                article_id_to_headline = {article.article_id: article.headline for article in req.candidates.articles}

                structured_output = {
                    "recommendations": [
                        {
                            "rank": idx + 1,
                            "headline": article.headline,
                            "original_headline": article_id_to_headline.get(article.article_id, "Unknown"),
                        }
                        for idx, article in enumerate(response.recommendations.articles)
                    ],
                    "profile_name": profile_path.stem,
                    "pipeline_name": pipeline_name,
                    "run_number": run_num,
                    "execution_time_seconds": execution_time,
                    "timestamp": time.time(),
                    "candidate_pool": [i.headline for i in req.candidates.articles],
                    "recommender_meta": response.recommender.model_dump() if response.recommender else None,
                }

                all_results.append(structured_output)

                # Save individual run results
                filename = f"data/{profile_path.stem}_{pipeline_name}_run{run_num}_output.json"
                with open(filename, "w") as output_file:
                    json.dump(structured_output, output_file, indent=2)

    # Save aggregated results with timing summary
    timing_summary = {}
    for result in all_results:
        key = (result["profile_name"], result["pipeline_name"])
        if key not in timing_summary:
            timing_summary[key] = []
        timing_summary[key].append(result["execution_time_seconds"])

    # Calculate timing statistics
    timing_stats = {}
    for (profile, pipeline), times in timing_summary.items():
        timing_stats[f"{profile}_{pipeline}"] = {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "runs": len(times),
        }

    # Save timing summary
    with open("data/timing_summary.json", "w") as timing_file:
        json.dump(timing_stats, timing_file, indent=2)

    # Save all results
    with open("data/all_results.json", "w") as all_results_file:
        json.dump(all_results, all_results_file, indent=2)

    num_profiles = len([p for p in (project_root_path / "tests" / "request_data" / "profiles").glob("*.json")])
    print(f"\nCompleted testing {len(available_pipelines)} pipelines on {num_profiles} profiles")
    print("Results saved to data/ directory")
