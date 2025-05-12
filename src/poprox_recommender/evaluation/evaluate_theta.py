import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def find_balanced_pairs(csv_path, output_path=None):
    # Load the data
    df = pd.read_csv(csv_path)

    # Define the metrics and their directions (1 for higher is better, -1 for lower is better)
    metrics = {
        "NDCG@10": 1,  # Higher is better
        "KL_TOPIC": -1,  # Lower is better
        "KL_LOC": -1,  # Lower is better
        # "event_level_prompt_ratio": 1,  # higher is better
        # "inside_loc_threshold": 1,  # Higher is better
        # "num_treatment": 1,  # Higher is better
    }

    # Normalize all metrics to [0,1] range
    scaler = MinMaxScaler()
    for metric in metrics.keys():
        df[f"{metric}_norm"] = scaler.fit_transform(df[[metric]])

    # Apply direction (invert metrics where lower is better)
    for metric, direction in metrics.items():
        if direction == -1:
            df[f"{metric}_norm"] = 1 - df[f"{metric}_norm"]

    # Create a combined score (can adjust weights here)
    weights = {
        "NDCG@10_norm": 0.4,
        "KL_TOPIC_norm": 0.3,
        "KL_LOC_norm": 0.3,
        # "event_level_prompt_ratio_norm": 0,
        # "inside_loc_threshold_norm": 0,
        # "num_treatment_norm": 0,
    }

    df["combined_score"] = sum(df[col] * weight for col, weight in weights.items())

    # Find top balanced pairs
    top_pairs = df.sort_values("combined_score", ascending=False)

    # Group by theta pairs and take the best from each group
    grouped = top_pairs.groupby(["theta_topic", "theta_loc"]).first().reset_index()
    final_results = grouped.sort_values("combined_score", ascending=False)

    # Select relevant columns for output
    output_cols = ["theta_topic", "theta_loc", "combined_score"] + list(metrics.keys())
    final_results = final_results[output_cols]

    if output_path:
        final_results.to_csv(output_path, index=False)

    return final_results


if __name__ == "__main__":
    input = "/workspaces/poprox-recommender-locality/outputs/metrics.csv"
    results = find_balanced_pairs(input)
    print(results.head(10))
