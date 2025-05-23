---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Offline Evaluation Metrics Visualizations

This notebook visualizes user-specific performance metrics of various recommenders in the mind-subset dataset to assess effectiveness and ranking overlap. We explore two metric groups:

1. **Effectiveness Metrics**: We use ranking-based metrics, Normalized Discounted Cumulative Gain (NDCG) and Reciprocal Rank (RR), to evaluate recommender effectiveness.
2. **Ranking Overlap Metrics**: We use Rank-Based Overlap (RBO) to assess consistency in top-k recommendations relative to final rankings.

> [!NOTE]
> This is a *parameterized* notebook, and is used to render the other specific evaluation notebooks with [Papermill](https://papermill.readthedocs.io/en/latest/).  By default, it displays
> the MIND Subset results, for easy editing.

```{code-cell} ipython3
:tags: [parameters]

EVAL_NAME = "mind-subset"
```

## Setup

+++

### Importing Libraries

+++

PyData packages:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
```

Local code and display support:

```{code-cell} ipython3
from IPython.display import HTML

from poprox_recommender.eval_tables import EvalTable
```

### Loading Data

```{code-cell} ipython3
mind_subset_user_metrics = pd.read_csv(f"../outputs/{EVAL_NAME}-profile-metrics.csv.gz")
mind_subset_user_metrics.head()
```

```{code-cell} ipython3
effectiveness_metrics = ["NDCG@5", "NDCG@10", "RR"]
overlap_metrics = ["RBO@5", "RBO@10"]
```

## Results

+++

### Effectiveness Metrics

NDCG measures how well the recommendations align with user test data, focusing on the top-k positions, such as the top 5 (NDCG@5) or top 10 (NDCG@10). Likewise, RR evaluates how well the recommender finds the most relevant item as the top result.

```{code-cell} ipython3
plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.3)

for i, metric in enumerate(effectiveness_metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(data=mind_subset_user_metrics, x="pipeline", y=metric)
    plt.xticks(rotation=45)

plt.show()
```

The summary tables show the mean values, standard deviation, and quantiles (10%ile, Median, 90%ile), each accompanied by their respective 95% confidence intervals for effectiveness metrics across recommenders.

```{code-cell} ipython3
for metric in effectiveness_metrics:
    tw = EvalTable(mind_subset_user_metrics, "pipeline", metric)
    tw.add_stat("Mean", np.mean, ci=True)
    tw.add_stat("Std Dev", np.std, ci=True)
    tw.add_quantiles(["10%ile", "Median", "90%ile"], [0.1, 0.5, 0.9], ci=True)
    display(HTML(f"<h3>Summary Table for {metric}</h3>"))
    display(HTML(tw.html_table()))
```

### Ranking Overlap Metrics
RBO measures the similarity between two ranked lists, evaluating how much overlap exists between pure top-k recommendations and the actual rankings produced after recommendations. RBO can be applied at different list depths to analyze performance consistency, such as RBO@5 and RBO@10.

```{code-cell} ipython3
plt.figure(figsize=(8, 3))
plt.subplots_adjust(wspace=0.3)

for i, metric in enumerate(overlap_metrics, 1):
    plt.subplot(1, 2, i)
    sns.barplot(data=mind_subset_user_metrics, x="pipeline", y=metric)
    plt.xticks(rotation=45)

plt.show()
```

The summary tables show the mean values, standard deviation, and quantiles (10%ile, Median, 90%ile), each accompanied by their respective 95% confidence intervals for ranking overlap metrics across recommenders.

```{code-cell} ipython3
for metric in overlap_metrics:
    tw = EvalTable(mind_subset_user_metrics, "pipeline", metric)
    tw.add_stat("Mean", np.mean, ci=True)
    tw.add_stat("Std Dev", np.std, ci=True)
    tw.add_quantiles(["10%ile", "Median", "90%ile"], [0.1, 0.5, 0.9], ci=True)
    display(HTML(f"<h3>Summary Table for {metric}</h3>"))
    display(HTML(tw.html_table()))
```
