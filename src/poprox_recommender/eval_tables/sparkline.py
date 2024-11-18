"""
Code for creating sparkline data.
"""

from io import BytesIO, StringIO

import matplotlib.pyplot as plt
import numpy as np

from .statutil import kde_max


class SparkKDEBase:
    def __init__(self, kernel, *, scale=None):
        self.kernel = kernel
        if scale is None:
            self.scale = kde_max(kernel)
        else:
            self.scale = scale


class SparkKDELatexWriter(SparkKDEBase):
    """
    Write a single KDE sparkline.
    """

    def __init__(self, kernel, *, file=None, **kwargs):
        super().__init__(kernel, **kwargs)

        if file is None:
            self.file = StringIO()
            self.file_owned = True
        else:
            self.file = file
            self.file_owned = False

    def begin(self, width):
        print("\\begin{sparkline}{%s}" % (width,), file=self.file)

    def point(self, x, color="red"):
        y = self.kernel(x)[0] / self.scale
        print("\\sparkdot %.3f %.3f %s" % (x, y, color), file=self.file)

    def render_kde(self, res=50):
        xs = np.linspace(0, 1, res)
        ys = self.kernel(xs)
        yss = ys / self.scale
        print("\\spark", file=self.file)
        for x, y in zip(xs, yss):
            print("  %.3f %.3f" % (x, y), file=self.file)
        print("  /", file=self.file)

    def end(self):
        print("\\end{sparkline}", file=self.file)

    def get_content(self):
        if not self.file_owned:
            raise RuntimeError("cannot get content w/ oputput file")
        return self.file.getvalue()


class SparkKDEImageWriter(SparkKDEBase):
    """
    Write a single KDE sparkline to PNG image data

    Key ideas for how to set up Matplotlib for this from https://github.com/crdietrich/sparklines/blob/master/sparklines.py.
    """

    ax: plt.Axes

    def __init__(self, kernel, **kwargs):
        super().__init__(kernel, **kwargs)

    def begin(self, width=4, height=0.25):
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis("off")

    def point(self, x, color="red"):
        y = self.kernel(x)[0] / self.scale
        self.ax.plot(x, y, marker="o", color=color)

    def render_kde(self, res=50):
        xs = np.linspace(0, 1, res)
        ys = self.kernel(xs)
        yss = ys / self.scale
        self.ax.plot(xs, yss, color="black")

    def end(self):
        self.fig.subplots_adjust(left=0)
        self.fig.subplots_adjust(right=0.99)
        self.fig.subplots_adjust(bottom=0.1)
        self.fig.subplots_adjust(top=0.9)
        bio = BytesIO()
        plt.savefig(bio)
        plt.close()
        self.img_bytes = bio.getvalue()

    def get_content(self):
        return self.img_bytes
