# THIS PLOT WAS SUPERSEEDED BY plot_102!!!
# Use that instead

from __future__ import annotations

from typing import Optional, Dict, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon

import plot_003
# import data_003
import data_007
# import data_005 as data
import data_101 as data_artificial
import probability_101 as prob
# from color import color_interpolate
from misc import region_from_positives


airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def fig_plot(  # noqa: C901
    z: float,
    tau: float,
    airspeed: int,
    right_limit: Optional[float] = None,
    save_as: Optional[str] = None,
    latex: bool = False,
    subplot_adjust_params: Optional[Dict[str, float]] = None,
    undertitle: bool = True,
    bars: bool = False,
    skip: int = 0,
    original_data: bool = False
) -> None:
    if save_as and latex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 10
        })
    else:
        matplotlib.use('Qt5Agg')

    if original_data:
        total, means, stds, n_stall = data_007.dists_given_airspeed(airspeed)
    else:
        AoAs, means, stds, n_stall = data_artificial.dists_given_airspeed(airspeed, skip=skip)

    # Plot parameters
    if right_limit is not None:
        right_plot = right_limit
    else:
        right_plot = (means + z*stds).max()

    shift_x = 1.1
    left_plot = 0
    plot_density = 1000

    ##################################
    # Plotting prep
    # xs = np.arange(left_plot, right_plot, 1)
    xs = np.arange(left_plot, right_plot*shift_x, (right_plot-left_plot)/plot_density)

    ##################################
    # Plotting classification results
    se = prob.wing_SE_classification(means, stds, n_stall, tau)

    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9), sharex='col', gridspec_kw={'height_ratios': [12, 12, 1.6]})
    # ax.set_xlabel('Signal Energy')
    ax.set_title(f"\\(v = \\) {airspeed} m/s" if latex else f"v = {airspeed} m/s")

    ax.set_ylabel('Angle of Attack')
    ax2.set_ylabel('P[stall | X=x]')
    ax3.set_ylabel('SE')
    if undertitle:
        ax3.set_xlabel(r'Signal Energy $(V^2 \cdot s)$' if latex
                       else 'Signal Energy (V^2 s)')

    # Plot 1
    # This line seems to not be necessary in the latest version of `prob` code,
    # it can be replaced with the data obtained from data_005
    # means_, stds_, _, _, stall_per_dist = data_003.dists_given_airspeed(airspeed, False)
    if bars or original_data:
        if 'total' not in vars():
            assert 'AoAs' in vars()
            plot_003.plot(ax, means, stds, int(AoAs.max()), z, n_stall,
                          yticks_sep=0, anglesofattack=AoAs)
        else:
            plot_003.plot(ax, means, stds, total, z, n_stall, yticks_sep=0,
                          start_at_zero=True)
    else:
        plot_top(ax, AoAs, means, stds, z, n_stall, yticks_sep=0)
    # ax.set_yticks([])

    # Plot 2
    ax2.plot(xs, np.vectorize(se.pdf_red)(xs), color='black')
    region2 = se.plot_certainty_region(ax2, xs, 0.09, '0.2', shift_y=0.13)

    # Plot 3
    region1_ = xs.reshape((-1, 1)) * np.ones((xs.size, means.size))
    means__ = means.reshape((1, -1))
    stds__ = stds.reshape((1, -1))
    region1_ = np.bitwise_and(means__-z*stds__ < region1_,
                              region1_ < means__+z*stds__)
    region1 = region1_[:, 0]
    for i in range(1, means.size):
        region1 = np.bitwise_or(region1, region1_[:, i])

    region = region1 * region2

    region_verts = region_from_positives(zip(xs, region))
    # region_verts = [(xs[0], 0), *zip(xs, region), (xs[-1], 0)]
    region_poly = Polygon(list(region_verts), facecolor='#60af3d', edgecolor='#fff')
    ax3.set_yticks([])
    ax3.add_patch(region_poly)
    ax3.set_ylim(bottom=0.03, top=1)

    if right_limit is not None:
        ax.set_xlim(right=right_limit)

    # from matplotlib.ticker import MaxNLocator
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.1, hspace=0, wspace=0)
    if subplot_adjust_params is not None:
        params = {'top': 1, 'bottom': 0.05, 'right': 1, 'left': 0.1, 'hspace': 0, 'wspace': 0}
        params.update(subplot_adjust_params)
        plt.subplots_adjust(**params)

    if save_as:
        plt.savefig(f'plots/{save_as}.png')
        if latex:
            plt.savefig(f'plots/{save_as}.pgf')
        else:
            plt.savefig(f'plots/{save_as}.svg')
        print(f'plots/{save_as}')
    else:
        plt.show()


def plot_top(
    ax: Any,
    anglesofattack: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    z: float,
    stall_per_dist: Iterable[float],
    right_limit: Optional[float] = None,
    yticks_sep: Optional[int] = 2,
) -> None:
    # z-confident region for each distribution
    poly = Polygon([*zip(means-z*stds, anglesofattack),
                    *reversed(list(zip(means+z*stds, anglesofattack)))])
    ax.add_patch(poly)
    # color_bars = [color_interpolate((0x46, 0x77, 0xc8), (0xc8, 0x5c, 0x46), d)
    #               for d in stall_per_dist]
    # ax.barh(anglesofattack, 2*z*stds, left=means-z*stds, height=.7,
    #         align='center', color=color_bars, zorder=2)

    # mean of distribution
    ax.plot(means, anglesofattack, color='#000000', zorder=4)

    # TODO: This is extremely lazy. It creates many boxes when one or a couple could be
    # enough
    # z-confident region given all distributions
    consistency_bar_size = (anglesofattack.max() - anglesofattack.min()) / 5.6
    ax.barh(np.zeros(stds.size)-.8*consistency_bar_size, 2*z*stds, left=means-z*stds,
            height=consistency_bar_size, align='center', color='#777', zorder=3)

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
    tau = .6
    z = 2

    # data_artificial.reload_global_data("data_Ahmed/Old Data")

    # fig_plot(z, tau, 6, undertitle=False, bars=True, original_data=True,
    #          save_as='plot_101/trying to find breaking z/plot_101_airspeed=6m_'
    #                  f'original_z={z}_tau={tau}')

    # Current problems with the plots:
    # - height of z-predictability
    # - dot size
    # - improve speed of classification (it was not a bottleneck in the past, it is now)
    for airspeed in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17]:
        fig_plot(z, tau, airspeed, undertitle=False, bars=True, original_data=True,
                 save_as=f'plot_101/plot_101_airspeed={airspeed}m_'
                         f'original_z={z}_tau={tau}')

        # for skip in [1000, 100]:
        #     fig_plot(z, tau, airspeed, undertitle=False, skip=skip, bars=True,
        #              save_as=f'plot_101/plot_101_airspeed={airspeed}m_'
        #                      f'precision={skip/1000}_z={z}_tau={tau}')
        # fig_plot(z, tau, airspeed, undertitle=False, bars=True, original_data=True)
