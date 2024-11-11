---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Offline Evaluation Metrics Visualizations
This notebook visualizes user-specific performance metrics of various recommenders in the MIND-small dataset to assess effectiveness and ranking overlap. We explore two metric groups:
1. **Effectiveness Metrics**: We use ranking-based metrics, Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR), to evaluate recommender effectiveness.
2. **Ranking Overlap Metrics**: We use Rank-Based Overlap (RBO) to assess consistency in top-k recommendations relative to fianl rankings.

+++

## 1. Setup

```{code-cell} ipython3
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t
from IPython.display import display, HTML

# Loading Data
mind_small_user_metrics = pd.read_csv('../outputs/mind-small-user-metrics.csv.gz')
mind_small_user_metrics.head()
```

## 2. Visualizations
To evaluate the plot, look for higher values in the bars. Higher values indicate better performance in terms of the respective metric. Error bars show the variability of the results. The exact values of the error bars are provided in the summary table below.

```{code-cell} ipython3
plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
sns.barplot(data=mind_small_user_metrics, x='recommender', y='NDCG@5')
plt.xticks(fontsize=8)

plt.subplot(1, 3, 2)
sns.barplot(data=mind_small_user_metrics, x='recommender', y='NDCG@10')
plt.xticks(fontsize=8)

plt.subplot(1, 3, 3)
sns.barplot(data=mind_small_user_metrics, x='recommender', y='MRR')
plt.xticks(fontsize=8)

plt.show()
```

```{code-cell} ipython3
# Function to calculate CI
def calculate_ci(data, confidence=0.95):
    se = sem(data)
    h = se * t.ppf((1 + confidence) / 2, len(data) - 1)
    return (np.mean(data) - h, np.mean(data) + h)

metrics = ['NDCG@5', 'NDCG@10', 'MRR']

html_tables = ""

# Generate summary tables
for metric in metrics:
    summary_table = mind_small_user_metrics.groupby('recommender')[metric].agg(
        Mean='mean'
    ).reset_index()

    summary_table['Mean (95% CI)'] = summary_table.apply(
        lambda row: (
            f"{row['Mean']:.4f}<br>"
            f"({calculate_ci(mind_small_user_metrics[mind_small_user_metrics['recommender'] == row['recommender']][metric])[0]:.4f} "
            f"- {calculate_ci(mind_small_user_metrics[mind_small_user_metrics['recommender'] == row['recommender']][metric])[1]:.4f})"
        ),
        axis=1
    )

    summary_table = summary_table[['recommender', 'Mean (95% CI)']]

    # Add to html_tables string for side by side display
    html_tables+= f"<div style='display: inline-block; margin-right: 50px;'><h3>Summary Table for {metric}</h3>{summary_table.to_html(index=False, escape=False)}</div>"

display(HTML(html_tables))
```

### 2. 2 Ranking Overlap Metrics

```{code-cell} ipython3
plt.figure(figsize=(8, 3))

plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 2, 1)
sns.barplot(data=mind_small_user_metrics, x='recommender', y='RBO@5')
plt.xticks(fontsize=8)

plt.subplot(1, 2, 2)
sns.barplot(data=mind_small_user_metrics, x='recommender', y='RBO@10')
plt.xticks(fontsize=8)

plt.show()
```

```{code-cell} ipython3
metrics = ['RBO@5', 'RBO@10']

html_tables = ""

for metric in metrics:
    summary_table = mind_small_user_metrics.groupby('recommender')[metric].agg(
        Mean='mean'
    ).reset_index()

    summary_table['Mean (95% CI)'] = summary_table.apply(
        lambda row: (
            f"{row['Mean']:.4f}<br>"
            f"({calculate_ci(mind_small_user_metrics[mind_small_user_metrics['recommender'] == row['recommender']][metric])[0]:.4f} "
            f"- {calculate_ci(mind_small_user_metrics[mind_small_user_metrics['recommender'] == row['recommender']][metric])[1]:.4f})"
        ),
        axis=1
    )

    summary_table = summary_table[['recommender', 'Mean (95% CI)']]

    html_tables+= f"<div style='display: inline-block; margin-right: 50px;'><h3>Summary Table for {metric}</h3>{summary_table.to_html(index=False, escape=False)}</div>"

display(HTML(html_tables))
```
