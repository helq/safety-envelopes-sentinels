from __future__ import annotations

from typing import Optional, Any, Dict, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
import data_003 as data
from misc import color_interpolate

import matplotlib


def fig_plot(
    z: int,
    airspeed: Optional[int],
    right_limit: Optional[float] = None,
    save_as: Optional[str] = None,
    subplot_adjust_params: Optional[Dict[str, float]] = None,
    only_png: bool = False,
    text: Optional[Tuple[float, float, str]] = None,
    yticks_sep: Optional[int] = 2,
    ylabel: Optional[str] = None
) -> None:
    if save_as:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 30
        })
    else:
        matplotlib.use('Qt5Agg')
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.size': 34
        })

    means, stds, total, n_stall, stall_per_dist = data.dists_given_airspeed(airspeed)

    ##################################
    # ## Plotting code
    ##################################
    f, ax = plt.subplots()
    # ax.set_yscale('log')
    plt.xlabel('Signal Energy')
    plt.ylabel('Angle of Attack' if ylabel is None else ylabel)

    plot(ax, means, stds, total, z, stall_per_dist, right_limit, yticks_sep)

    # plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.1, hspace=0, wspace=0)
    if subplot_adjust_params is not None:
        params = {'top': 1, 'bottom': 0.1, 'right': 1, 'left': 0.08, 'hspace': 0, 'wspace': 0}
        params.update(subplot_adjust_params)
        plt.subplots_adjust(**params)

    if text:
        ax.text(*text)

    # from matplotlib.ticker import MaxNLocator
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    if save_as:
        if not only_png:
            plt.savefig(f'plots/{save_as}.pgf')
        plt.savefig(f'plots/{save_as}.png')
    else:
        plt.show()


def plot(
    ax: Any,
    means: np.ndarray,
    stds: np.ndarray,
    total: int,
    z: float,
    stall_per_dist: Iterable[float],
    right_limit: Optional[float] = None,
    yticks_sep: Optional[int] = 2,
    anglesofattack: Optional[np.ndarray] = None,
    start_at_zero: bool = False
) -> None:
    if anglesofattack is None:
        anglesofattack = np.arange(0, stds.size) if start_at_zero else \
                         np.arange(1, stds.size+1)

    # ax.barh(np.ones(stds.size)*(stds.size+1)/2, 2*z*stds, left=means-z*stds,
    #         height=stds.size+.5, align='center', color='#9cbfe3', zorder=1)
    color_bars = [color_interpolate((0x46, 0x77, 0xc8), (0xc8, 0x5c, 0x46), d)
                  for d in stall_per_dist]
    ax.barh(anglesofattack, 2*z*stds, left=means-z*stds, height=.7,
            align='center', color=color_bars, zorder=2)
    consistency_bar_size = total / 5.6
    ax.barh(np.zeros(stds.size)-.8*consistency_bar_size, 2*z*stds, left=means-z*stds,
            height=consistency_bar_size, align='center', color='#777', zorder=3)
    ax.scatter(means, anglesofattack, color='#000000', zorder=4)
    if yticks_sep is None:
        ax.set_yticks([])
    elif yticks_sep > 0:
        ax.set_yticks(anglesofattack[yticks_sep-1::yticks_sep])
        ax.set_yticklabels(anglesofattack[yticks_sep-1::yticks_sep])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-.8*consistency_bar_size)
    if right_limit is not None:
        ax.set_xlim(right=right_limit)
    # ax.barh(np.arange(1, 19), [150], left=[50], height=.5, align='center')


if __name__ == '__main__':
    data.legacy_data(revert_to_bad=True)

    # fig_plot(1, 0)

    # fig_plot(1, 0, subplot_adjust_params={'left': 0.13, 'bottom': 0.15})
    # fig_plot(4, 0, subplot_adjust_params={'left': 0, 'bottom': 0.18})
    # fig_plot(20, 0, subplot_adjust_params={'left': 0})

    fig_plot(1, 0, subplot_adjust_params={'left': 0.15, 'bottom': 0.18}, save_as='plot_003_airspeed_6ms_m1')  # noqa
    # fig_plot(4, 0, right_limit=890, subplot_adjust_params={'left': 0, 'bottom': 0.18}, save_as='plot_003_airspeed_6ms_m4')  # noqa
    # fig_plot(20, 0, subplot_adjust_params={'left': 0, 'bottom': 0.18}, save_as='plot_003_airspeed_6ms_m20')  # noqa
    # fig_plot(12, save_as='plot_003_airspeed_20m')
    # fig_plot(None, save_as='plot_003_all_airspeeds')
    # fig_plot(None, right_limit=9000, save_as='plot_003_all_airspeeds_capped')

    for i, m in enumerate(np.arange(0.1, 10.1, 0.1)):
        subplot_params = {'left': 0.15, 'bottom': 0.18, 'right': 0.97, 'top': 0.97}
        # fig_plot(m, 0, subplot_adjust_params=subplot_params,
        #          right_limit=1060,
        #          save_as=f'animation-1/6ms/{i:03d}-plot_003_airspeed_6ms_m={m:.1f}',
        #          only_png=True)

        # fig_plot(m, 12, subplot_adjust_params=subplot_params,
        #          right_limit=11000,
        #          save_as=f'animation-1/20ms/{i:03d}-plot_003_airspeed_20ms_m={m:.1f}',
        #          only_png=True, yticks_sep=1,
        #          text=(6000, 3, f"z = {m:.1f}"))

        fig_plot(m, None, subplot_adjust_params=subplot_params,
                 right_limit=9000,
                 save_as=f'animation-1/all/{i:03d}-plot_003_all_airspeeds_m={m:.1f}',
                 only_png=True, yticks_sep=None, ylabel="AoA and airspeeds")

        # break
