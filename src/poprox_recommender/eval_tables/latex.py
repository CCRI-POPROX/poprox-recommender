from io import StringIO

import numpy as np
from pylatex import escape_latex

from .sparkline import SparkKDELatexWriter


def write_latex(cols, rows, sys_label, scale):
    layout = "l" + "c" * len(cols) + "c"
    out = StringIO()

    print("\\begin{tabular}{%s}" % (layout,), file=out)
    print("%s" % (sys_label,), file=out)
    for stat in cols:
        print("& %s" % (escape_latex(stat),), file=out)
    print("& Dist. (KDE) \\\\", file=out)
    print("\\toprule", file=out)

    for row in rows:
        print("%s &" % (escape_latex(row.name),), file=out)
        for sr in row.stats:
            if sr.low is None:
                print("%.3f &" % (sr.value,), file=out)
            else:
                print("\\begin{tabular}{@{}c@{}}%", file=out)
                print("%.3f \\\\%%" % (sr.value,), file=out)
                print("{\\smaller[2] (%.3f, %.3f)}\\end{tabular} &" % (sr.low, sr.high), file=out)

        w = SparkKDELatexWriter(row.kde, file=out, scale=scale)
        w.begin(9)
        w.point(np.mean(row.data))
        w.render_kde()
        w.end()

        print("\\\\", file=out)

    print("\\bottomrule", file=out)
    print("\\end{tabular}", file=out)

    return out.getvalue()
