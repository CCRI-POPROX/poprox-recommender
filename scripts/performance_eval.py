import json
from pathlib import Path

import pandas as pd

metadata_files = Path("./data/pipeline_outputs").glob("**/metadata.json")

all_llm_metrics = []
for metadata_file in metadata_files:
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    llm_metrics = metadata["llm_metrics"]
    user_profile_metrics = llm_metrics["ranker"]["user_profile_generation"]
    article_ranking_metrics = llm_metrics["ranker"]["article_ranking"]
    rewriter_metrics = llm_metrics["rewriter"]

    # Prefix user_profile and article_ranking keys
    user_profile_metrics_prefixed = {f"user_profile_{k}": v for k, v in user_profile_metrics.items()}
    article_ranking_metrics_prefixed = {f"article_ranking_{k}": v for k, v in article_ranking_metrics.items()}

    # Prefix rewriter metrics and compute averages
    rewriter_metrics_prefixed = {}
    if rewriter_metrics:
        input_tokens = [m.get("input_tokens", 0) for m in rewriter_metrics]
        output_tokens = [m.get("output_tokens", 0) for m in rewriter_metrics]
        durations = [m.get("duration_seconds", 0) for m in rewriter_metrics]
        rewriter_metrics_prefixed["rewriter_avg_input_tokens"] = sum(input_tokens) / len(input_tokens) if input_tokens else 0
        rewriter_metrics_prefixed["rewriter_avg_output_tokens"] = sum(output_tokens) / len(output_tokens) if output_tokens else 0
        rewriter_metrics_prefixed["rewriter_avg_duration_seconds"] = sum(durations) / len(durations) if durations else 0

    # Combine all metrics
    all_llm_metrics.append({
        **user_profile_metrics_prefixed,
        **article_ranking_metrics_prefixed,
        **rewriter_metrics_prefixed
    })

df = pd.DataFrame(all_llm_metrics)

averages = {}
for col in df.columns:
    if any(key in col for key in ["duration_seconds", "input_tokens", "output_tokens"]):
        averages[f"avg_{col}"] = df[col].mean()

print("Averages:")
for k, v in averages.items():
    print(f"{k}: {v}")
