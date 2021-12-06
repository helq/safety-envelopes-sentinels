from __future__ import annotations

from typing import Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import data_003 as data
# import probability_001 as prob
from math import sqrt
from matplotlib.collections import PatchCollection

import matplotlib


def fig_plot(
    sample_size: int,
    gamma_mean: int,
    gamma_std: int,
    airspeed: Optional[int],
    save_as: Optional[str] = None
) -> None:
    if save_as:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    else:
        matplotlib.use('Qt5Agg')

    means, stds, total, n_stall, stall_per_dist = data.dists_given_airspeed(airspeed)

    # https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound#Normal_variance_with_known_mean  # noqa
    E_stds = sqrt(2 / sample_size) * stds**2

    print("Estimated standard deviation for a sample of size 10")
    print(E_stds)

    # ## Plotting code
    ##################################
    # plt.figure(figsize=(6, 8))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax.set_yscale('log')
    ax.set_ylabel('Standard deviation of signal')
    ax.set_xlabel('Angle of Attack')

    anglesofattack = np.arange(1, stds.size+1)
    # ax.barh(np.ones(stds.size)*(stds.size+1)/2, 2*gamma_std*stds, left=means-gamma_std*stds,
    #         height=stds.size+.5, align='center', color='#9cbfe3')
    # ax.barh(anglesofattack, 2*gamma_std*E_stds, left=stds-gamma_std*E_stds, height=.7,
    #         align='center', color='#4677c8', zorder=0)

    # low_E_stds = stds**2-gamma_std*E_stds
    # low_E_stds[low_E_stds < 0] = 0
    # low_E_stds = np.sqrt(low_E_stds)
    # up_E_stds = np.sqrt(stds**2+gamma_std*E_stds)
    # height_E_stds = up_E_stds - low_E_stds
    low_E_stds = stds**2-gamma_std*E_stds
    up_E_stds = stds**2+gamma_std*E_stds
    height_E_stds = up_E_stds - low_E_stds
    color_bars = ['#4677c8' if d == 0 else '#c85c46' for d in stall_per_dist]
    ax.bar(anglesofattack, height_E_stds, bottom=low_E_stds,  # height=.7,
           align='center', color=color_bars, zorder=0)
    # ax.bar(2*gamma_std*E_stds, anglesofattack, bottom=stds-gamma_std*E_stds,  # width=.7,
    #        align='center', color='#4677c8', zorder=0)
    # ax.barh(np.zeros(stds.size)-.2, 2*gamma_std*stds, left=means-gamma_std*stds,
    #         height=.3, align='center', color='#31538c')
    # ax.scatter(anglesofattack, stds, color='#000000', zorder=4)
    ax.scatter(anglesofattack, stds**2, color='#000000', zorder=4)
    ax.set_xticks(anglesofattack)
    ax.set_xticklabels(anglesofattack)
    ax.set_ylim(bottom=0, top=max(stds**2)*1.5)

    ax2.set_yscale('log')
    ax2.set_ylabel('Variance of signal energy')
    ax2.set_xlabel('Signal energy')

    make_error_boxes(ax2, means, stds**2, gamma_mean*stds, low_E_stds, up_E_stds)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0, top=max(stds**2)*1.5)

    if save_as:
        plt.savefig(f'plots/{save_as}.pgf')
        plt.savefig(f'plots/{save_as}.png')
    else:
        plt.show()


def make_error_boxes(
    ax: Any,
    xdata: np.ndarray,
    ydata: np.ndarray,
    xerror: np.ndarray,
    y_low: np.ndarray,
    y_up: np.ndarray,
    facecolor: str = '#9cbfe3',
    edgecolor: str = 'None',
    alpha: float = 0.8
) -> None:
    # print(list(zip(xdata, ydata, xerror, yerror)))
    # Loop over data points; create box from errors at each point
    errorboxes = [matplotlib.patches.Rectangle((x - xe, yl), 2*xe, yu-yl)
                  for x, y, xe, yl, yu in zip(xdata, ydata, xerror, y_low, y_up)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    ax.errorbar(xdata, ydata, xerr=xerror,
                yerr=np.array([ydata-y_low, y_up-ydata]), fmt='None',
                ecolor='k', alpha=0.5)


if __name__ == '__main__':
    data.legacy_data(revert_to_bad=True)

    fig_plot(sample_size=10, gamma_mean=4, gamma_std=2, airspeed=0)

    # fig_plot(10, 4, 2, 0, save_as='plot_004_airspeed_6m')
    # fig_plot(10, 4, 2, 12, save_as='plot_004_airspeed_20m')
    # fig_plot(10, 4, 2, None, save_as='plot_004_all_airspeeds')
