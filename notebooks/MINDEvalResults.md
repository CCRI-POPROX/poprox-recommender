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
This notebook visualizes user-specific performance metrics of various recommenders in the MIND validation dataset to assess effectiveness and ranking overlap. We explore two metric groups:
1. **Effectiveness Metrics**: We use ranking-based metrics, Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR), to evaluate recommender effectiveness.
2. **Ranking Overlap Metrics**: We use Rank-Based Overlap (RBO) to assess consistency in top-k recommendations relative to fianl rankings.

+++

## 1. Setup

+++

### 1. 1 Importing Libraries

+++

PyData packages:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t
```

```{code-cell} ipython3
from lkdemo.plotutils import *
from eval_tables import EvalTable
from IPython.display import HTML
```

Set up progress and logging output:

```{code-cell} ipython3
from tqdm.auto import tqdm
import logging
import lenskit.util

_log = logging.getLogger('notebook')
lenskit.util.log_to_notebook()
tqdm.pandas()
```

Save Output

```{code-cell} ipython3
fig_dir = init_figs('Results')
print(fig_dir)
```

### 1. 2 Loading Data

```{code-cell} ipython3
mind_val_user_metrics = pd.read_csv('../outputs/mind-val-user-metrics.csv.gz')
mind_val_user_metrics.head()
```

## 2. Results
To evaluate the plot, look for higher values in the bars. Higher values indicate better performance in terms of the respective metric. Error bars show the variability of the results. The exact values of the error bars are provided in the summary table below.

+++

### 2. 1 Effectiveness Metrics

```{code-cell} ipython3
plt.figure(figsize=(12, 3))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 3, 1)
sns.barplot(data=mind_val_user_metrics, x='recommender', y='NDCG@5')
plt.xticks(fontsize=8)

plt.subplot(1, 3, 2)
sns.barplot(data=mind_val_user_metrics, x='recommender', y='NDCG@10')
plt.xticks(fontsize=8)

plt.subplot(1, 3, 3)
sns.barplot(data=mind_val_user_metrics, x='recommender', y='MRR')
plt.xticks(fontsize=8)

plt.show()
```

```{code-cell} ipython3
tw = EvalTable(mind_val_user_metrics, 'recommender', 'NDCG@5', progress=tqdm)
tw.add_stat('Mean', np.mean, ci=True)
tw.add_quantiles(['10%ile', 'Median', '90%ile'], [0.1, 0.5, 0.9], ci=True)
tw_fn = fig_dir / 'cf-example.tex'
```

```{code-cell} ipython3
tw_fn.write_text(tw.latex_table())
HTML(tw.html_table())
```

### 2. 2 Ranking Overlap Metrics

```{code-cell} ipython3
plt.figure(figsize=(8, 3))

plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 2, 1)
sns.barplot(data=mind_val_user_metrics, x='recommender', y='RBO@5')
plt.xticks(fontsize=8)

plt.subplot(1, 2, 2)
sns.barplot(data=mind_val_user_metrics, x='recommender', y='RBO@10')
plt.xticks(fontsize=8)

plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
