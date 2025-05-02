import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.linalg import LinAlgError

from . import statutil as u

_log = logging.getLogger(__name__)


class RowData(NamedTuple):
    name: object
    data: np.ndarray
    kde: sps.gaussian_kde | None
    stats: list


class EvalTable:
    def __init__(self, table, sys_col, metric_col, sys_label="Algorithm", progress=None):
        self.table = table
        self.sys_col = sys_col
        self.metric_col = metric_col
        self.sys_label = sys_label
        self.progress = progress if progress else lambda x, **kwargs: x
        self.stats = []

        self._prepare_table()

    def _prepare_table(self):
        self.rowdata = []
        for g, gs in self.table.groupby(self.sys_col)[self.metric_col]:
            try:
                kde = sps.gaussian_kde(gs)
            except LinAlgError:
                kde = None
            self.rowdata.append(RowData(g, np.array(gs), kde, []))

        self.kde_scale = np.max([u.kde_max(r.kde) for r in self.rowdata if r.kde is not None])  # type: ignore

    def add_stat(self, label, func, ci=True):
        self.stats.append(label)
        for rd in self.progress(self.rowdata, desc=label):
            sr = u.estimate_stat(rd.data, func, ci=ci)
            rd.stats.append(sr)

    def add_quantiles(self, labels, quantiles, ci=True):
        assert len(labels) == len(quantiles)
        self.stats += labels
        for rd in self.progress(self.rowdata, desc="quantiles"):
            sr = u.boot_quantiles(rd.data, quantiles, ci=ci)
            for r in sr:
                rd.stats.append(r)

    def dataframe(self):
        return pd.DataFrame.from_records(
            [[s.value for s in rd.stats] for rd in self.rowdata],
            index=[rd.name for rd in self.rowdata],
            columns=self.stats,
        )

    def latex_table(self):
        from .latex import write_latex

        return write_latex(self.stats, self.rowdata, self.sys_label, self.kde_scale)

    def html_table(self):
        from .html import write_html

        return write_html(self.stats, self.rowdata, self.sys_label, self.kde_scale)
