from __future__ import annotations

from enum import Enum
from math import isinf
from matplotlib import cm, ticker
from scipy.stats import norm
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import portion as P
import sys

import data_104 as data_experiments
import data_101 as data_artificial

from typing import List, Tuple, Optional, TextIO, Any

from misc import Plot, horizontal_interval, plot_horizontal_interval, \
    intervals_from_true, find_plot_points

dirname = 'plots/plot_106'
inf = float('inf')

Interval = Tuple[float, float]


class State(Enum):
    NoStall = 0
    Stall = 1


class SafetyEnvelope(object):
    def __init__(
        self,
        stall_means: np.ndarray,
        stall_stds: np.ndarray,
        nostall_means: np.ndarray,
        nostall_stds: np.ndarray,
    ):
        self.stall_means = stall_means
        self.stall_stds = stall_stds
        self.nostall_means = nostall_means
        self.nostall_stds = nostall_stds

    def compute_metrics(
        self,
        nostall_intervals: P.Interval,
        stall_intervals: P.Interval,
        w: float = 10
    ) -> Tuple[float, float, float, float]:
        coverage = accuracy = error = quality = 0.0
        if not nostall_intervals.empty:
            for inter in nostall_intervals:
                area = norm.cdf(inter.upper, loc=self.nostall_means, scale=self.nostall_stds) \
                    - norm.cdf(inter.lower, loc=self.nostall_means, scale=self.nostall_stds)
                accuracy += float(area.sum())

                area = norm.cdf(inter.upper, loc=self.stall_means, scale=self.stall_stds) \
                    - norm.cdf(inter.lower, loc=self.stall_means, scale=self.stall_stds)
                error += float(area.sum())

        if not stall_intervals.empty:
            for inter in stall_intervals:
                area = norm.cdf(inter.upper, loc=self.nostall_means, scale=self.nostall_stds) \
                    - norm.cdf(inter.lower, loc=self.nostall_means, scale=self.nostall_stds)
                error += float(area.sum())

                area = norm.cdf(inter.upper, loc=self.stall_means, scale=self.stall_stds) \
                    - norm.cdf(inter.lower, loc=self.stall_means, scale=self.stall_stds)
                accuracy += float(area.sum())

        total_intervals = len(self.nostall_means) + len(self.stall_means)
        accuracy /= total_intervals
        error /= total_intervals
        coverage = accuracy + error
        quality = accuracy * (1-error)**w
        # quality = w*error + (1-(error + accuracy))
        return coverage, accuracy, error, quality

    def raw_intervals(
        self, z: float
    ) -> Tuple[np.ndarray, List[State]]:
        intervals = []
        for m, s in zip(self.nostall_means, self.nostall_stds):
            intervals.append((float(m - z*s), float(m + z*s)))
        for m, s in zip(self.stall_means, self.stall_stds):
            intervals.append((float(m - z*s), float(m + z*s)))

        states = [State.NoStall]*len(self.nostall_means) + [State.Stall]*len(self.stall_means)

        return np.array(intervals), states


class LogicalSE(SafetyEnvelope):
    def __init__(
        self,
        stall_means: np.ndarray,
        stall_stds: np.ndarray,
        nostall_means: np.ndarray,
        nostall_stds: np.ndarray,
    ):
        super().__init__(stall_means, stall_stds, nostall_means, nostall_stds)

    def stall_nostall_intervals(
        self, z: float
    ) -> Tuple[P.interval, P.interval]:
        intervals, states = self.raw_intervals(z)
        stall_intervals = P.empty()
        nostall_intervals = P.empty()
        for inter, st in zip(intervals, states):
            if st == State.Stall:
                stall_intervals |= P.closed(inter[0], inter[1])
            else:
                nostall_intervals |= P.closed(inter[0], inter[1])

        return nostall_intervals, stall_intervals

    def se_intervals(
        self, z: float
    ) -> Tuple[P.interval, P.interval]:
        nostall_intervals, stall_intervals = self.stall_nostall_intervals(z)
        new_nostall = nostall_intervals - stall_intervals
        new_stall = stall_intervals - nostall_intervals

        return new_nostall, new_stall


class ConditionalSE(SafetyEnvelope):
    def __init__(
        self,
        stall_means: np.ndarray,
        stall_stds: np.ndarray,
        nostall_means: np.ndarray,
        nostall_stds: np.ndarray,
    ):
        super().__init__(stall_means, stall_stds, nostall_means, nostall_stds)
        self.stall_prob: Optional[np.ndarray] = None
        self._stall_prob_parametrs = (0.0, 0.0, 0.0)

    def predictable_intervals(self, z: float) -> P.Interval:
        intervals, states = self.raw_intervals(z)
        interval = P.empty()
        for inter in intervals:
            interval |= P.closed(inter[0], inter[1])
        return interval

    def compute_stall_prob(
        self,
        xs: np.ndarray
    ) -> np.ndarray:
        ys_stall = norm.pdf(xs.reshape((1, -1)),
                            loc=self.stall_means, scale=self.stall_stds)
        ys_nostall = norm.pdf(xs.reshape((1, -1)),
                              loc=self.nostall_means, scale=self.nostall_stds)
        red_prop = ys_stall.sum(axis=0)
        blue_prop = ys_nostall.sum(axis=0)
        return red_prop / (red_prop + blue_prop)  # type: ignore

    def _compute_stall_prob(
        self,
        xs: np.ndarray
    ) -> np.ndarray:
        if self.stall_prob is not None and (
            self._stall_prob_parametrs == (xs[0], xs[-1], len(xs))
        ):
            stall_prob = self.stall_prob
        else:
            stall_prob = self.compute_stall_prob(xs)
            self.stall_prob = stall_prob
        return stall_prob

    def compute_conditional_intervals(
        self, xs: np.ndarray, tau: float
    ) -> Tuple[P.Interval, P.Interval]:
        stall_prob = self._compute_stall_prob(xs)

        stall_intervals = P.empty()
        nostall_intervals = P.empty()
        for left, right in intervals_from_true(zip(xs, stall_prob < 1 - tau)):
            nostall_intervals |= P.closed(left, right)
        for left, right in intervals_from_true(zip(xs, stall_prob > tau)):
            stall_intervals |= P.closed(left, right)

        return nostall_intervals, stall_intervals


class ConditionalSE_GPRM_extended(ConditionalSE):
    def __init__(
        self,
        stall_means: np.ndarray,
        stall_stds: np.ndarray,
        nostall_means: np.ndarray,
        nostall_stds: np.ndarray,
        gprm_means: np.ndarray,
        gprm_stds: np.ndarray,
        AoAs: np.ndarray,
    ):
        super().__init__(stall_means, stall_stds, nostall_means, nostall_stds)
        assert len(gprm_means) == len(gprm_stds)
        self.gprm_means = gprm_means
        self.gprm_stds = gprm_stds
        self.AoAs = AoAs

    def raw_intervals(
        self, z: float
    ) -> Tuple[np.ndarray, List[State]]:
        intervals = [(float(m - z*s), float(m + z*s))
                     for m, s in zip(self.gprm_means, self.gprm_stds)]

        states = [State.NoStall]*len(self.gprm_means)

        return np.array(intervals), states


def compute_xlim_plot(
    intervals: Any, xlim: Tuple[float, float]
) -> Tuple[float, float]:
    """It finds the smallest and biggest values in intervals and sets them as the
    limits, if the initial interval was infinite"""
    left, right = xlim
    intervals_ = np.array(intervals)
    left_inf = isinf(left) and left < 0
    right_inf = isinf(right) and right > 0
    left = float(intervals_.min()) if left_inf else left
    right = float(intervals_.max()) if right_inf else right
    cushion = (right - left) / 80
    if left_inf:
        left -= cushion
    if right_inf:
        right += cushion
    return left, right


def figure_logic_SE(  # noqa: C901
    logicalSE: LogicalSE,
    header: str,
    z: float,
    xlim: Tuple[float, float] = (-inf, inf),
    ylim: Tuple[float, float] = (0, inf),
    resolution: int = 300,
    no_bells: bool = False,
    discard_negative: bool = True,
    show_AoAs: bool = True,
    no_ylabels_n_ticks: bool = False,
    w: float = 10
) -> Tuple[plt.Figure, plt.Axes]:

    left, right = xlim
    if no_bells:
        fig, ax1 = plt.subplots(1, 1, sharex='col', figsize=(5, 3))
        ax2 = ax1
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(5, 4.5))

    intervals, is_stall = logicalSE.raw_intervals(z)

    left, right = compute_xlim_plot(intervals, xlim)

    xs = np.linspace(left, right, resolution)

    # Plot 1 - Bells
    if not no_bells:
        ys_red = norm.pdf(xs.reshape((1, -1)),
                          loc=logicalSE.stall_means,
                          scale=logicalSE.stall_stds)
        ys_blue = norm.pdf(xs.reshape((1, -1)),
                           loc=logicalSE.nostall_means,
                           scale=logicalSE.nostall_stds)

        upper = 2*float(ys_red.max()) if isinf(ylim[1]) else ylim[1]
        ax1.set_ylim(bottom=ylim[0], top=upper)

        # Small optimization to delete all the points that fall outside of what is being
        # shown
        upper_p_cushion = (upper-ylim[0])*1.02
        ys_red[ys_red > upper_p_cushion] = upper_p_cushion
        ys_blue[ys_blue > upper_p_cushion] = upper_p_cushion

        for ys_ in ys_red:
            ax1.plot(xs, ys_, color='red')
        for ys_ in ys_blue:
            ax1.plot(xs, ys_, color='blue')

        blue_accuracies = []
        red_accuracies = []
        i, j = 0, 0
        for inter, st in zip(intervals, is_stall):
            if st == State.NoStall:
                blue_accuracy = ys_blue[i].copy()
                blue_accuracy[xs < float(inter[0])] = 0
                blue_accuracy[float(inter[1]) < xs] = 0
                blue_accuracies.append(blue_accuracy)
                i += 1
            else:
                red_accuracy = ys_red[j].copy()
                red_accuracy[xs < float(inter[0])] = 0
                red_accuracy[float(inter[1]) < xs] = 0
                red_accuracies.append(red_accuracy)
                j += 1

        for blue_accuracy in blue_accuracies:
            ax1.fill_between(xs, blue_accuracy, alpha=.3, color='blue')
        for red_accuracy in red_accuracies:
            ax1.fill_between(xs, red_accuracy, alpha=.3, color='red')

    # Plot 2
    for i, (inter, st) in enumerate(zip(intervals, is_stall)):
        ml = horizontal_interval([inter], ["#49f" if st == State.NoStall else "#f94"],
                                 yshift=i, capsize=.9)
        ax2.add_collection(ml)

    top = len(intervals) + 1

    total_lines = 3
    ybottom = (top / 25) * total_lines
    capsize = - ybottom / total_lines
    # nostall_ypos = - ybottom * 3/(total_lines*2 + 1)
    # stall_ypos = - ybottom * 5/(total_lines*2 + 1)
    grey_ypos = - ybottom * 3/(total_lines*2 + 1)
    nostall_se_ypos = - ybottom * 5/(total_lines*2 + 1)
    stall_se_ypos = - ybottom * 5/(total_lines*2 + 1)

    # nostall_intervals, stall_intervals = logicalSE.stall_nostall_intervals(z)

    # plot_horizontal_interval(ax2, nostall_intervals, "blue",
    #                          yshift=nostall_ypos, capsize=capsize)
    # plot_horizontal_interval(ax2, stall_intervals, "red",
    #                          yshift=stall_ypos, capsize=capsize)

    plot_horizontal_interval(ax2, P.open(xlim[0], xlim[1]), "#AAA",
                             yshift=grey_ypos, dashes='dashed')

    nostall_se_intervals, stall_se_intervals = logicalSE.se_intervals(z)
    # Making sure that these don't go over the negative number line:
    if discard_negative:
        nostall_se_intervals &= P.open(0, inf)
        stall_se_intervals &= P.open(0, inf)

    plot_horizontal_interval(ax2, stall_se_intervals, "red",
                             yshift=stall_se_ypos, capsize=capsize,
                             assume_closed=True)
    plot_horizontal_interval(ax2, nostall_se_intervals, "blue",
                             yshift=nostall_se_ypos, capsize=capsize,
                             assume_closed=True)

    ax2.set_ylim(top=top, bottom=-ybottom)
    ax2.set_xlim(left=left, right=right)

    coverage, accuracy, error, quality = \
        logicalSE.compute_metrics(nostall_se_intervals, stall_se_intervals, w=w)

    # Labels and other informative stuff
    ax1.set_title(f"{header}\n" +
                  f"\\(z\\)={z} \\(coverage\\)={coverage*100:.3f}%\n"
                  f"\\(accuracy\\)={accuracy*100:.3f}% "
                  f"\\(error\\)={error*100:.3f}%\n"
                  f"\\(quality\\)={quality*100:.3f}%")
    ax2.set_xlabel(r"Signal Energy ($V^2 \cdot s$)")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if no_ylabels_n_ticks:
        ax1.set_yticks([])
        ax2.set_yticks([])
    else:
        if show_AoAs:
            ax2.set_ylabel("Angle of attack")
        else:
            ax2.set_ylabel("Configuration\ndistribution")
            ax2.set_yticks([])

    return fig, ax1


def figure_conditional_SE(
    conditionalSE: ConditionalSE,
    header: str,
    z: float,
    tau: float,
    xlim: Tuple[float, float] = (-inf, inf),
    dthreshold: float = 0.001,
    partitions: List[int] = [40, 5],
    discard_negative: bool = True,
    show_AoAs: bool = True,
    no_ylabels: bool = False,
    w: float = 10
) -> Tuple[plt.Figure, plt.Axes]:

    fig, (ax1, ax2, ax3) = \
        plt.subplots(3, 1, sharex='col', figsize=(5, 4.5),
                     gridspec_kw={'height_ratios': [3, 2, .6]})

    intervals, is_stall = conditionalSE.raw_intervals(z)

    left, right = compute_xlim_plot(intervals, xlim)

    # xs = np.linspace(left, right, resolution)
    # This threshold function forces a better resolution closer to 0 or 1
    ythreshold = lambda x: 0.02*np.exp(-((x-0.5)*4.2)**2)  # noqa: E731
    xs = find_plot_points(conditionalSE.compute_stall_prob,
                          float(intervals.min()), float(intervals.max()),
                          dthreshold=dthreshold, ythreshold=ythreshold,
                          partitions=partitions)
    print(f"xs.size = {len(xs)}")

    # Plot 1 - "Bells"
    if isinstance(conditionalSE, ConditionalSE_GPRM_extended):
        intervals_gprm, _ = conditionalSE.raw_intervals(z)
        lines = [((inter[0], i), (inter[1], i))
                 for i, inter in zip(conditionalSE.AoAs, intervals_gprm)]
        ax1.add_collection(mc.LineCollection(lines, colors='grey'))
        top = float(conditionalSE.AoAs.max()) + 1

        intervals, is_stall = SafetyEnvelope.raw_intervals(conditionalSE, z)

    for i, (inter, st) in enumerate(zip(intervals, is_stall)):
        ml = horizontal_interval([inter], ["#49f" if st == State.NoStall else "#f94"],
                                 yshift=i, capsize=.9)
        ax1.add_collection(ml)
    if 'top' not in locals():
        top = len(intervals)

    total_lines = 3
    ybottom = (top / 25) * total_lines
    capsize = - ybottom / total_lines
    # nostall_ypos = - ybottom * 3/(total_lines*2 + 1)
    # stall_ypos = - ybottom * 5/(total_lines*2 + 1)
    grey_ypos = - ybottom * 3/(total_lines*2 + 1)
    stall_se_ypos = - ybottom * 5/(total_lines*2 + 1)

    plot_horizontal_interval(ax1, P.open(left, right), "#AAA",
                             yshift=grey_ypos, dashes='dashed')

    pred_intervals = conditionalSE.predictable_intervals(z)

    plot_horizontal_interval(ax1, pred_intervals, "green",
                             yshift=stall_se_ypos, capsize=capsize,
                             assume_closed=True)

    ax1.set_ylim(top=top, bottom=-ybottom)
    ax1.set_xlim(left=left, right=right)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Plot 2 - tau-confidence
    nostall_intervals, stall_intervals = \
        conditionalSE.compute_conditional_intervals(xs, tau)

    # Plotting using a xs with less points
    xs_ = find_plot_points(conditionalSE.compute_stall_prob, left, right,
                           dthreshold=0.005)
    # ax2.scatter(xs_, conditionalSE.compute_stall_prob(xs_), color="black", marker='.')
    ax2.plot(xs_, conditionalSE.compute_stall_prob(xs_), color="black")

    plot_horizontal_interval(ax2, stall_intervals, "#f94",
                             yshift=-.08, capsize=0.07, assume_closed=True)
    plot_horizontal_interval(ax2, nostall_intervals, "#49f",
                             yshift=-.08, capsize=0.07, assume_closed=True)

    plot_horizontal_interval(ax2, P.open(xlim[0], xlim[1]), "#AAA",
                             yshift=0, dashes='dashed')

    ax2.set_ylim(bottom=-.16)

    # Plot 3 - Safety Envelopes
    nostall_se_intervals = nostall_intervals & pred_intervals
    stall_se_intervals = stall_intervals & pred_intervals

    # Making sure that these don't go over the negative number line:
    if discard_negative:
        nostall_se_intervals &= P.open(0, inf)
        stall_se_intervals &= P.open(0, inf)

    # Computing metrics
    coverage, accuracy, error, quality = \
        conditionalSE.compute_metrics(nostall_se_intervals, stall_se_intervals, w=w)

    plot_horizontal_interval(ax3, stall_se_intervals, "red",
                             yshift=0, capsize=0.07, assume_closed=True)
    plot_horizontal_interval(ax3, nostall_se_intervals, "blue",
                             yshift=0, capsize=0.07, assume_closed=True)
    ax3.set_ylim(bottom=-.1, top=.1)
    ax3.set_yticks([])

    # Labels and other informative stuff
    ax1.set_title(f"{header}\n" +
                  f"\\(z\\)={z} \\(\\tau\\)={tau} \\(coverage\\)={coverage*100:.3f}%\n"
                  f"\\(accuracy\\)={accuracy*100:.3f}% "
                  f"\\(error\\)={error*100:.3f}%\n"
                  f"\\(quality\\)={quality*100:.3f}%")
    ax3.set_xlabel(r"Signal Energy ($V^2 \cdot s$)")
    if not no_ylabels:
        ax3.set_ylabel("SE")
        ax2.set_ylabel(r"$P(stall \mid X=x)$")
        ax1.set_ylabel("Angle of attack" if show_AoAs else "Configuration\ndistribution")
    if not show_AoAs:
        ax1.set_yticks([])

    return fig, ax1


def figure_logic_SE_exploration(
    logicalSE: LogicalSE,
    header: str,
    zs: np.ndarray,
    w: float = 10,
    notes_file: Optional[TextIO] = None,
    show_legend: bool = True
    # xlim: Tuple[float, float]
) -> Tuple[plt.Figure, plt.Axes]:
    # left, right = xlim
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    coverage = []
    accuracy = []
    error = []
    quality = []
    for z in zs:
        nostall_intervals, stall_intervals = logicalSE.se_intervals(z)
        # Making sure that these don't go over the negative number line:
        nostall_intervals &= P.open(0, inf)
        stall_intervals &= P.open(0, inf)

        cov, acc, err, com = \
            logicalSE.compute_metrics(nostall_intervals, stall_intervals, w=w)
        coverage.append(cov)
        accuracy.append(acc)
        error.append(err)
        quality.append(com)

    ax.plot(zs, coverage, color='black', label='coverage')
    ax.plot(zs, accuracy, color='blue', label='accuracy')
    ax.plot(zs, error, color='red', label='error')
    ax.plot(zs, quality, color='green', label='quality')
    if show_legend:
        ax.legend()

    argmax_cov = int(np.argmax(coverage))
    argmax_acc = int(np.argmax(accuracy))
    argmax_err = int(np.argmax(error))
    argmax_com = int(np.argmax(quality))

    ax.scatter([zs[argmax_cov]], [coverage[argmax_cov]], marker='x', color='black')
    ax.scatter([zs[argmax_acc]], [accuracy[argmax_acc]], marker='x', color='blue')
    ax.scatter([zs[argmax_err]], [error[argmax_err]], marker='x', color='red')
    ax.scatter([zs[argmax_com]], [quality[argmax_com]], marker='x', color='green')

    ax.set_title(header)
    ax.set_xlabel(r"\(z\)")

    notes_file = sys.stdout if notes_file is None else notes_file
    print(f"Max for quality z = {zs[argmax_com]}", file=notes_file)
    print(f"Max for quality coverage = {coverage[argmax_com]}", file=notes_file)
    print(f"Max for quality accuracy = {accuracy[argmax_com]}", file=notes_file)
    print(f"Max for quality error    = {error[argmax_com]}", file=notes_file)
    print(f"Max for quality quality =  {quality[argmax_com]}", file=notes_file)

    return fig, ax


class Metric(Enum):
    Coverage = 0
    Accuracy = 1
    Error = 2
    Quality = 3


def figure_conditional_SE_exploration(  # noqa: C901
    conditionalSE: ConditionalSE,
    header: str,
    zs: np.ndarray,
    pseudo_taus: np.ndarray,
    dthreshold: float = 0.001,
    partitions: List[int] = [40, 5],
    w: float = 10,
    metric: Metric = Metric.Quality,
    notes_file: Optional[TextIO] = None,
    discard_negative: bool = True,
    no_title_metric: bool = False
) -> None:
    # taus = 1 - (1 - (norm.cdf(pseudo_taus) - norm.cdf(-pseudo_taus)))/2
    taus = norm.cdf(pseudo_taus)

    intervals, is_stall = conditionalSE.raw_intervals(zs.max())
    intervals_ = np.array(intervals)
    left = float(intervals_.min())
    right = float(intervals_.max())

    # xs = np.linspace(left, right, resolution)
    # This threshold function forces a better resolution closer to 0 or 1
    ythreshold = lambda x: 0.02*np.exp(-((x-0.5)*4.2)**2)  # noqa: E731
    xs = find_plot_points(conditionalSE.compute_stall_prob, left, right,
                          dthreshold=dthreshold, ythreshold=ythreshold,
                          partitions=partitions)
    print(f"xs.size = {len(xs)}")
    if len(xs) < 100:
        print("WARNING: the size of `xs` is less than 100. "
              "Expect innaccurate metric computations.")

    # Computing SE intervals
    pred_intervalss = []
    for z in zs:
        pred_intervalss.append(conditionalSE.predictable_intervals(z))
    conditional_intervalss = []
    for tau in taus:
        conditional_intervalss.append(conditionalSE.compute_conditional_intervals(xs, tau))

    coverage = np.zeros((len(zs), len(taus)))
    accuracy = np.zeros((len(zs), len(taus)))
    error = np.zeros((len(zs), len(taus)))
    quality = np.zeros((len(zs), len(taus)))

    for i, pred_intervals in enumerate(pred_intervalss):
        for j, (nostall_intervals, stall_intervals) in enumerate(conditional_intervalss):
            nostall_se_intervals = nostall_intervals & pred_intervals
            stall_se_intervals = stall_intervals & pred_intervals

            # Making sure that these don't go over the negative number line:
            if discard_negative:
                nostall_se_intervals &= P.open(0, inf)
                stall_se_intervals &= P.open(0, inf)

            # Computing metrics
            cov, acc, err, com = \
                conditionalSE.compute_metrics(nostall_se_intervals, stall_se_intervals, w=w)

            coverage[i, j] = cov
            accuracy[i, j] = acc
            error[i, j] = err
            quality[i, j] = com

    if metric == Metric.Coverage:
        metric_vals = coverage
        argmax = coverage.argmax()
        maxval = coverage.flatten()[argmax]
    elif metric == Metric.Accuracy:
        metric_vals = accuracy
        argmax = accuracy.argmax()
        maxval = accuracy.flatten()[argmax]
    elif metric == Metric.Error:
        metric_vals = error
        argmax = error.argmin()
        maxval = error.flatten()[argmax]
    elif metric == Metric.Quality:
        metric_vals = quality
        argmax = quality.argmax()
        maxval = quality.flatten()[argmax]

    argmax_z = argmax // len(taus)
    argmax_tau = argmax % len(taus)
    max_z = zs[argmax_z]
    max_tau = taus[argmax_tau]

    # Plotting
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    pseudo_taus, zs = np.meshgrid(pseudo_taus, zs)
    ax.plot_surface(pseudo_taus, zs, metric_vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '0.999..' if x > 3.4 else f"{norm.cdf(x):.3f}"))

    ax.set_ylabel(r'\(z\)')
    ax.set_xlabel(r'\(\tau\)')
    # ax.set_title(f"{metric.name} - {header}\n"
    metric_head = "" if no_title_metric else f"{metric.name}\n"
    if metric == Metric.Error:
        ax.set_title(metric_head +
                     f"minimum={maxval:.3f}\nwith \\(z\\)={max_z:.3f} \\(\\tau\\)={max_tau:.4f}")
    else:
        ax.set_title(metric_head +
                     f"maximum={maxval:.3f}\nwith \\(z\\)={max_z:.3f} \\(\\tau\\)={max_tau:.4f}")
    # airspeed_str = 'all airspeeds' if airspeed is None else f"airspeed={airspeed}"
    # ax.set_title(f"{metric.name} for {airspeed_str}\n"
    #              f"maximum={maxval:.3f} with \\(z\\)={max_z:.3f} \\(\\tau\\)={max_tau:.4f}")

    notes_file = sys.stdout if notes_file is None else notes_file
    print(f"{metric.name}\n"
          f"z={max_z}\ntau={max_tau}\n"
          f"Coverage (for max) = {coverage[argmax_z, argmax_tau]}\n"
          f"Accuracy (for max) = {accuracy[argmax_z, argmax_tau]}\n"
          f"Error (for max)    = {error[argmax_z, argmax_tau]}\n"
          f"Quality (for max) =  {quality[argmax_z, argmax_tau]}",
          file=notes_file)


def get_conditionalSE(airspeed: Optional[int]) -> ConditionalSE:
    total, means, vars_, stall = \
        data_experiments.dists_given_airspeed(airspeed)
    stds = np.sqrt(vars_)

    means_blue = means[stall == 0].reshape((-1, 1))
    stds_blue = stds[stall == 0].reshape((-1, 1))
    means_red = means[stall == 1].reshape((-1, 1))
    stds_red = stds[stall == 1].reshape((-1, 1))

    return ConditionalSE(means_red, stds_red, means_blue, stds_blue)


def get_logicalSE(airspeed: Optional[int]) -> LogicalSE:
    total, means, vars_, stall = \
        data_experiments.dists_given_airspeed(airspeed)
    stds = np.sqrt(vars_)

    means_blue = means[stall == 0].reshape((-1, 1))
    stds_blue = stds[stall == 0].reshape((-1, 1))
    means_red = means[stall == 1].reshape((-1, 1))
    stds_red = stds[stall == 1].reshape((-1, 1))

    return LogicalSE(means_red, stds_red, means_blue, stds_blue)


def get_gprmExtendedSE(airspeed: int) -> ConditionalSE_GPRM_extended:
    total, means, vars_, stall = \
        data_experiments.dists_given_airspeed(airspeed)
    stds = np.sqrt(vars_)

    means_blue = means[stall == 0].reshape((-1, 1))
    stds_blue = stds[stall == 0].reshape((-1, 1))
    means_red = means[stall == 1].reshape((-1, 1))
    stds_red = stds[stall == 1].reshape((-1, 1))

    AoAs, gprm_means, gprm_vars, gprm_stall = \
        data_artificial.dists_given_airspeed(airspeed, skip=int(1000*density))
    gprm_stds = np.sqrt(gprm_vars)

    return ConditionalSE_GPRM_extended(
        means_red, stds_red, means_blue, stds_blue,
        gprm_means, gprm_stds, AoAs
    )


if False and __name__ == '__main__':
    dirname += '/logicalSE-plots'

    airspeed = 6
    logSE = get_logicalSE(airspeed)
    # for z in [0.1, 1.2, 1.88, 1.9]:
    first = True
    for z in [0.1, 1.2, 1.9]:
        # with Plot(dirname, f"logicalSE-airspeed={airspeed}-z={z}"):
        with Plot():
            header = f"Airspeed = {airspeed}"
            figure_logic_SE(logicalSE=logSE, header=header, z=z, xlim=(0, 1.6),
                            no_ylabels_n_ticks=not first)
        first = False

    logSE = get_logicalSE(None)
    header = "All airspeeds"
    first = True
    for z in [0.1, 0.5, 0.865]:
        # with Plot(dirname, f"logicalSE-ll_airspeeds-z={z}"):
        with Plot():
            figure_logic_SE(logicalSE=logSE, header=header, z=z, xlim=(0, 95),
                            no_bells=True, show_AoAs=False,
                            no_ylabels_n_ticks=not first)
        first = False

if False and __name__ == '__main__':
    dirname += '/conditionalSE-plots'
    airspeed = 15
    condSE = get_conditionalSE(airspeed)
    header = f"Airspeed = {airspeed}"
    for tau in [0.5, 0.8, 0.99]:
        for z in [0.4, 1, 2.8]:
            with Plot(dirname, f'conditionalSE-airspeed={airspeed}-tau={tau}-z={z}'):
                figure_conditional_SE(
                    conditionalSE=condSE, header=header, z=z, tau=tau, xlim=(0, 120))

    condSE = get_conditionalSE(None)
    header = "All airspeeds"
    for tau in [0.5, 0.8, 0.99]:
        for z in [0.4, 1, 2.8]:
            with Plot(dirname, f'conditionalSE-all_airspeeds-tau={tau}-z={z}'):
                figure_conditional_SE(
                    conditionalSE=condSE, header=header, z=z, tau=tau, xlim=(0, 130),
                    show_AoAs=False
                )

if False and __name__ == '__main__':
    dirname += '/conditionalSE-plots'
    for z, tau in [(1, 0.8), (2, .99)]:
        first = True
        for airspeed in [6, 20]:
            condSE = get_conditionalSE(airspeed)
            header = f"Airspeed {airspeed} m/s"
            with Plot(dirname, f'conditionalSE-airspeed={airspeed}-tau={tau}-z={z}'):
                figure_conditional_SE(
                    conditionalSE=condSE, header=header, z=z, tau=tau, xlim=(0, inf),
                    no_ylabels=not first)
            first = False

        # condSE = get_conditionalSE(None)
        # header = "All airspeeds"
        # with Plot(dirname, f'conditionalSE-all_airspeeds-tau={tau}-z={z}'):
        #     figure_conditional_SE(
        #         conditionalSE=condSE, header=header, z=z, tau=tau, xlim=(0, inf),
        #         show_AoAs=False, no_ylabels=True)

if False and __name__ == '__main__':
    dirname += '/logicalSE-exploration'
    w = 10
    # w = 30
    # for airspeed in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    for airspeed in [6, 20]:
        logSE = get_logicalSE(airspeed)
        header = f"Airspeed = {airspeed}"
        # with Plot(dirname, f"airspeed={airspeed}-w={w}", font_size=22) as notesf:
        with Plot() as notesf:
            figure_logic_SE_exploration(
                logicalSE=logSE, header=header,
                zs=np.linspace(0, 3, 250), notes_file=notesf, w=w, show_legend=False)

    logSE = get_logicalSE(None)
    header = "All airspeeds"
    # with Plot(dirname, f"all-airspeeds-w={w}", font_size=22) as notesf:
    with Plot() as notesf:
        figure_logic_SE_exploration(
            logicalSE=logSE, header=header, zs=np.linspace(0, 1.2, 250),
            notes_file=notesf, w=w)

if True and __name__ == '__main__':
    dirname += '/conditionalSE-exploration'
    w = 10
    # w = 30

    condSE = get_conditionalSE(None)
    header = "All airspeeds"
    pseudo_taus = np.linspace(0, 4, 30)
    zs = np.linspace(0, 0.45, 20)
    # for metric in Metric:
    for metric in [Metric.Accuracy, Metric.Error, Metric.Quality]:
        # with Plot() as notesf:
        with Plot(dirname, f"all-airspeeds-{metric.name}-w={w}", font_size=20) as notesf:
            figure_conditional_SE_exploration(
                conditionalSE=condSE, header=header,
                zs=zs, pseudo_taus=pseudo_taus, metric=metric,
                notes_file=notesf, w=w,
                no_title_metric=True
            )

    pseudo_taus = np.linspace(0, 3, 20)
    zs = np.linspace(0, 4, 20)
    # for airspeed in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    # for airspeed in [6, 20]:
    for airspeed in [20]:
        condSE = get_conditionalSE(airspeed)
        header = f"airspeed = {airspeed}"
        # for metric in Metric:
        for metric in [Metric.Accuracy, Metric.Error, Metric.Quality]:
            # with Plot() as notesf:
            with Plot(dirname, f"airspeed={airspeed}-{metric.name}-w={w}",
                      font_size=20) as notesf:
                figure_conditional_SE_exploration(
                    conditionalSE=condSE, header=header,
                    zs=zs, pseudo_taus=pseudo_taus, metric=metric,
                    w=w, partitions=[40, 6], notes_file=notesf,
                    no_title_metric=True
                )

if False and __name__ == '__main__':
    dirname += '/conditionalSE-exploration/finding-more-precise-taus'

    # pseudo_taus = np.linspace(.9, 1.7, 300)  # when w = 10
    pseudo_taus = np.linspace(1.2, 2.8, 300)  # when w = 30
    zs = np.linspace(3, 4, 2)
    # w = 10
    w = 30
    for airspeed in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        condSE = get_conditionalSE(airspeed)
        header = f"airspeed = {airspeed}"
        # with Plot() as notesf:
        with Plot(dirname, f"airspeed={airspeed}-Quality-w={w}") as notesf:
            figure_conditional_SE_exploration(
                conditionalSE=condSE, header=header,
                zs=zs, pseudo_taus=pseudo_taus, metric=Metric.Quality,
                w=w, partitions=[40, 6], notes_file=notesf)

    condSE = get_conditionalSE(None)
    header = "All airspeeds"
    # with Plot() as notesf:
    with Plot(dirname, f"all-airspeeds-Quality-w={w}") as notesf:
        figure_conditional_SE_exploration(
            conditionalSE=condSE, header=header,
            zs=zs, pseudo_taus=pseudo_taus, metric=Metric.Quality,
            notes_file=notesf, w=w
        )

# Checking how `w` affects the best tau.
# The short answer is:
# * big w's make tau big, small w's make tau small
# * I can't answer yet what is the relationship between the two
#  (This is because the biggest accuracy will always happen with a small tau (around 0.5),
#  (which also causes the biggest error), the smallest error is obtained when tau is big.)
# * Could there be an analytical formula to show the relationship between tau and
#   accuracy, and tau and error? (assuming a sufficiently large z, something like 4 or 5)
# * Also, there is a value of z that rules out tau (but this is dependent on the
#   distributions. If the distributions are too close, tau wins in the war. If the
#   distributions are far apart, z wins the war)
# * The real
if False and __name__ == '__main__':
    pseudo_taus = np.linspace(0, 3, 200)
    zs = np.linspace(0, 4.4, 13)
    w = 10

    blue = np.array([[-6], [1]])
    red = np.array([[3], [1.4]])

    means_blue, stds_blue = blue.reshape((2, -1, 1))
    means_red, stds_red = red.reshape((2, -1, 1))
    header = "Custom distribution"
    condSE = ConditionalSE(means_red, stds_red, means_blue, stds_blue)
    logSE = LogicalSE(means_red, stds_red, means_blue, stds_blue)

    with Plot():
        figure_conditional_SE(
            conditionalSE=condSE,
            header=header, z=4, tau=.9987)
        figure_logic_SE(
            logicalSE=logSE, header=header, z=2
        )
        figure_conditional_SE_exploration(
            conditionalSE=condSE,
            header=header, zs=zs, pseudo_taus=pseudo_taus,
            metric=Metric.Quality, discard_negative=False,
            w=w, partitions=[40, 6])

if False and __name__ == '__main__':
    dirname += '/optimal-quality-SE-plots'

    airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # zs_logic = [1.77108, 0.60241, 2.00000, 1.91566, 1.39759, 1.66265, 1.63855,
    #             2.08434, 0.66265, 2.07229, 2.14458, 2.03614, 0.66265]  # for w=10

    # for airspeed, z in zip(airspeeds, zs_logic):
    #     header = f"Airspeed = {airspeed}"
    #     with Plot(f'{dirname}/logic', f'airspeed={airspeed}-z={z}'):
    #         figure_logic_SE(logicalSE=get_logicalSE(airspeed), header=header,
    #                         z=z, xlim=(0, inf), resolution=100)

    taus = [0.9079264, 0.9043408, 0.9083675, 0.9079264, 0.9052467, 0.9070394,
            0.9052467, 0.9070395, 0.9056973, 0.9052467, 0.9052467, 0.9061463,
            0.9034286]  # for w=10

    for airspeed, tau in zip(airspeeds, taus):
        if airspeed not in {6, 20}:
            continue
        header = f"Airspeed {airspeed} m/s"
        z = 4.0
        with Plot(f'{dirname}/conditional', f'airspeed={airspeed}-z={z}-tau={tau}'):
            figure_conditional_SE(
                conditionalSE=get_conditionalSE(airspeed), header=header,
                z=z, tau=tau, xlim=(0, inf))

    # z = 0.525301
    # with Plot(f'{dirname}/logic', f'all_airspeeds-z={z}'):
    #     header = "All airspeeds"
    #     figure_logic_SE(logicalSE=get_logicalSE(None), header=header,
    #                     z=z, xlim=(0, 70), no_bells=True, show_AoAs=False)

    z = 4.0
    tau = 0.8785076
    with Plot(f'{dirname}/conditional', f'all_airspeeds-z={z}-tau={tau}'):
        condSE = get_conditionalSE(None)
        header = "All airspeeds"
        figure_conditional_SE(conditionalSE=condSE, header=header,
                              z=z, tau=tau, xlim=(0, 70), show_AoAs=False)

if False and __name__ == '__main__':
    dirname += '/GPRM-extended'
    # airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # for airspeed in airspeeds:
    for airspeed, xlim_right in [(14, inf), (17, 12.1)]:
        # xlim_right = inf
        density = 0.1
        z = 0.3
        tau = 0.9

        # with Plot():
        with Plot(dirname, f"se-airspeed={airspeed}-z={z}-tau={tau}-density={density}"):
            condSE = get_conditionalSE(airspeed)
            figure_conditional_SE(
                conditionalSE=condSE,
                header=f"Safety envelopes (airspeed {airspeed} m/s)",
                z=z, tau=tau, xlim=(0, xlim_right))

        # with Plot():
        with Plot(dirname, f"gprm-extended-airspeed={airspeed}-z={z}-tau={tau}-density={density}"):
            gprmSE = get_gprmExtendedSE(airspeed)
            figure_conditional_SE(
                conditionalSE=gprmSE,
                header=f"GPRM extended safety envelopes (airspeed {airspeed} m/s)",
                z=z, tau=tau, xlim=(0, xlim_right))
