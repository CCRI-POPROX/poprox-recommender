"""
Utilities for working with kernel density estimators.
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import bootstrap


class StatCI(NamedTuple):
    """
    A statistic value with a confidence interval.
    """

    value: float
    low: float
    high: float


def estimate_stat(data, stat, ci=True, **kwargs):
    """
    Estimate a statistic and its upper & lower confidence intervals.
    Wraps :fun:`scipy.stats.bootstrap`.

    Args:
        data(array-like):
            The array of values to summarize.
        stat:
            The statistic to compute.
        kwargs:
            Additional arguments to :fun:`~scipy.stats.bootstrap`.

    Returns:
        StatCI: The results of the statistic with its confidence interval.
    """
    val = stat(data)
    if not ci:
        return StatCI(val, None, None)

    # vectorization & BCA both impose massive memory costs
    br = bootstrap([data], stat, method="basic", vectorized=False, **kwargs)
    ci = br.confidence_interval
    return StatCI(val, ci.low, ci.high)


def boot_quantiles(data, quants, NBOOT=9999, ci=True):
    n = len(data)
    nqs = len(quants)
    vals = np.quantile(data, quants)

    if not ci:
        return [StatCI(vals[i], None, None) for i in range(nqs)]

    stats = np.empty((NBOOT, nqs))
    for i in range(NBOOT):
        boot_samp = np.random.choice(data, size=n, replace=True)
        stats[i, :] = np.quantile(boot_samp, quants)

    qs = np.quantile(stats, [0.025, 0.975], axis=0)
    return [StatCI(vals[i], qs[0, i], qs[1, i]) for i in range(nqs)]


def kde_max(kde, full_result=False):
    """
    Find the maximum density of a kernel density estimator.

    Args:
        kde(scipy.stats.gaussian_kde):
            The kernel density estimator to analyze.
        full_result(bool):
            If ``True``, return the full :class:`~scipy.optimize.OptimizeResult` instead of just
            the maximum density.
    """

    def obj(x):
        return -kde(x)[0]

    res = minimize_scalar(obj)
    if full_result:
        res.fun = -res.fun
        return res
    else:
        return -res.fun
