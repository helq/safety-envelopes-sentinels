from __future__ import annotations

import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.collections as mc
import numpy as np
import os

import portion as P

from typing import Tuple, Iterator, Optional, Any, List, Callable, Union, TextIO


class Plot(object):
    def __init__(self,
                 dirname: str = "plots",
                 namefile: Optional[str] = None,
                 font_size: int = 14):
        self.dirname = dirname
        self.namefile = namefile
        self.font_size = font_size
        if namefile is not None:
            os.makedirs(dirname, exist_ok=True)

    def __enter__(self) -> Optional[TextIO]:
        if self.namefile is None:
            return None
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': self.font_size
        })
        self.notes_path = f'{self.dirname}/{self.namefile}-notes.txt'
        self.notes = open(self.notes_path, 'w')
        return self.notes

    def __exit__(self, exc_type: Any, value: Any, traceback: Any) -> None:
        # Delete "notes" file if nothing had been written to it
        if self.namefile:
            pos = self.notes.tell()
            self.notes.close()
            if pos == 0:  # nothing was written in file (probably)
                os.remove(self.notes_path)

        # If there was an error at execution time, don't plot just output the error
        # immediately
        if exc_type is not None:
            return

        if self.namefile:
            plt.savefig(f'{self.dirname}/{self.namefile}.png', bbox_inches='tight')
            plt.savefig(f'{self.dirname}/{self.namefile}.pgf', bbox_inches='tight')
            print(f'Plots saved to: `{self.dirname}/{self.namefile}`')
        else:
            plt.show()

        plt.close()


def color_interpolate(color1: Tuple[int, int, int], color2: Tuple[int, int, int], w: float) -> str:
    """
    Nice, gamma corrected interpolation
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    # Regular, old, ugly linear interpolation
    # r = r1 + w * (r2 - r1)
    # g = g1 + w * (g2 - g1)
    # b = b1 + w * (b2 - b1)

    r1_, g1_, b1_, r2_, g2_, b2_ = r1**2.2, g1**2.2, b1**2.2, r2**2.2, g2**2.2, b2**2.2
    r_ = r1_ + w * (r2_ - r1_)
    g_ = g1_ + w * (g2_ - g1_)
    b_ = b1_ + w * (b2_ - b1_)
    r, g, b = int(r_**(1/2.2)), int(g_**(1/2.2)), int(b_**(1/2.2))
    return f"#{r:02x}{g:02x}{b:02x}"


def region_from_positives(
    data: Iterator[Tuple[float, float]],
    low: float = 0.0
) -> Iterator[Tuple[float, float]]:
    last = next(data)
    if last[1] != low:
        yield (last[0], low)
    # yield (last[0], low)
    yield last
    height = last[1]
    for val in data:
        if val[1] != height:
            yield last
            yield (last[0], val[1])
            height = val[1]
        last = val
    if last[1] != low:
        yield last
    yield (last[0], low)


def intervals_from_true(
    data: Iterator[Tuple[float, bool]]
) -> Iterator[Tuple[float, float]]:
    left: Optional[float] = None
    last: float
    for x, v in data:
        if v:
            if left is None:
                left = x
                last = x
            else:
                last = x
        else:
            if left is not None:
                yield left, last
                left = None
    if left is not None:
        yield left, last


def horizontal_interval(
    intervals: List[Tuple[float, float]],
    colors: List[str],
    yshift: float = -.02,
    capsize: float = .02,
) -> mc.LineCollection:
    capsize /= 2
    return mc.LineCollection(
        list(itertools.chain(*[
            [((interval[0], yshift-capsize), (interval[0], yshift+capsize)),
             ((interval[0], yshift), (interval[1], yshift)),
             ((interval[1], yshift-capsize), (interval[1], yshift+capsize))]
            for interval in intervals])),
        colors=[color for color in colors for _ in range(3)])


def plot_horizontal_interval(
    ax: plt.Axes,
    intervals: P.interval,
    color: str,
    yshift: float = -.02,
    capsize: float = .02,
    dashes: str = 'solid',
    assume_closed: bool = False
) -> mc.LineCollection:
    if intervals.empty:
        return
    capsize /= 2
    lines = []
    dashess = []
    for inter in intervals:
        lines.append(((inter.lower, yshift), (inter.upper, yshift)))
        dashess.append(dashes)
        if assume_closed or inter.left == P.CLOSED:
            lines.append(((inter.lower, yshift-capsize), (inter.lower, yshift+capsize)))
            dashess.append('solid')
        if assume_closed or inter.right == P.CLOSED:
            lines.append(((inter.upper, yshift-capsize), (inter.upper, yshift+capsize)))
            dashess.append('solid')
    ml = mc.LineCollection(lines, colors=[color]*len(lines), dashes=dashess)
    ax.add_collection(ml)
    return ml


def find_plot_points(
    f: Callable[[np.ndarray], np.ndarray],
    bot: float,
    top: float,
    delta: float = 0.001,
    dthreshold: float = 0.01,
    ythreshold: Union[None, float, Callable[[np.ndarray], np.ndarray]] = None,
    partitions: List[int] = [40, 10]
) -> np.ndarray:
    assert bot < top
    assert len(partitions) > 0
    xs = np.linspace(bot, top, partitions[0])
    sep = xs[1] - xs[0]
    delta_ = delta * sep
    ys = f(xs)
    dys = (ys - f(xs - delta_)) / delta_
    to_zoom_in = np.abs((ys[:-1] + sep*dys[:-1]) - ys[1:]) > dthreshold
    # print(f"ys = {ys}")
    # print(f"dys = {dys}")
    if isinstance(ythreshold, float):
        zoom_in_y = np.abs(ys[:-1] - ys[1:]) > ythreshold
        to_zoom_in = to_zoom_in.__or__(zoom_in_y)
    elif ythreshold is not None:  # ie, is callable
        zoom_in_y = np.abs(ys[:-1] - ys[1:]) > ythreshold(ys[:-1])
        to_zoom_in = to_zoom_in.__or__(zoom_in_y)
    # print(f"to_zoom_in = {to_zoom_in}")
    new_xs = []
    for i, zoom in enumerate(to_zoom_in):
        if zoom:
            xs_zoomed = find_plot_points(
                f, xs[i], xs[i+1], delta=delta, dthreshold=dthreshold,
                ythreshold=ythreshold,
                partitions=partitions[1:] if len(partitions) > 1 else partitions)
            new_xs.append(xs_zoomed[:-1])
        else:
            new_xs.append(xs[i:i+1])
    new_xs.append(xs[-1:])
    return np.concatenate(new_xs)  # type: ignore


if __name__ == '__main__':
    f: Callable[[np.ndarray], np.ndarray]
    # from scipy.stats import norm
    # f = lambda x: 1 - (1 - (norm.cdf(x) - norm.cdf(-x)))/2  # type: ignore # noqa: E731
    f = lambda x: np.sin(x)  # type: ignore # noqa: E731
    # f = lambda x: 0.1*np.exp(-((x-0.5)*5)**2)  # type: ignore # noqa: E731
    # f = lambda x: 1 / (1 + np.exp(-x))  # type: ignore # noqa: E731
    # f = lambda x: x**3  # type: ignore # noqa: E731
    # xs = find_plot_points(f, 0, 4, dthreshold=0.01)
    # xs = find_plot_points(f, 0, 1, dthreshold=0.001)
    xs = find_plot_points(f, -10, 10, dthreshold=0.01, partitions=[40, 3])
    # xs = np.linspace(-10, 10, 1000)

    # ythreshold = lambda x: 0.02*np.exp(-((x-0.5)*4.2)**2)  # noqa: E731
    # xs = find_plot_points(f, -10, 10, ythreshold=ythreshold, partitions=3)

    print(f"xs.size = {len(xs)}")
    plt.plot(xs, f(xs))
    plt.scatter(xs, f(xs), marker='.')
    plt.show()
