from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import plot_003
import data_003
import data_005 as data
import probability_005 as prob

import matplotlib


airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def fig_plot(
    gamma: int,
    confidence: float,
    airspeed: Optional[int],
    right_limit: Optional[float] = None,
    save_as: Optional[str] = None,
    subplot_adjust_params: Optional[Dict[str, float]] = None,
    undertitle: bool = True
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

    means, stds, total, n_stall = data.dists_given_airspeed(airspeed)

    # Plot parameters
    if right_limit is not None:
        right_plot = right_limit
    else:
        maxi = np.where(means == means.max())[0].flatten()[0]
        right_plot = means[maxi] + gamma*stds[maxi]
    shift_x = 1.1
    left_plot = 0
    plot_density = 1000

    ##################################
    # Plotting prep
    # xs = np.arange(left_plot, right_plot, 1)
    xs = np.arange(left_plot, right_plot*shift_x, (right_plot-left_plot)/plot_density)

    ##################################
    # Plotting classification results

    # Original Safety Envelopes
    # se = prob.SafetyEnvelopes(means_1, stds_1, means_2, stds_2, 2)
    # Vanilla Safety Envelopes
    # se = prob.VanillaSafetyEnvelopes(means_1, stds_1, means_2, stds_2, 1.5)

    # Statistical Inference (with each distribution as mixture distribution)
    # se = prob.StatisticalInference(means_1, stds_1, means_2, stds_2, 0.95)

    # Statistical Inference influenced by Safety Envelopes
    # se = prob.univariate_SISE(means_1, stds_1, means_2, stds_2, 0.5, 1.5)
    # se2 = prob.univariate_SISE(means_1, stds_1, means_2, stds_2, 0.95, 3)

    se = prob.wing_BIWT(means, stds, total-n_stall, confidence, 0)
    # Enumerating probability of no-stall given a distribution
    # print([(i, rgd.p) for i, rgd in enumerate(se.red_given_dist)])

    # PLOTTING
    ##################################
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex='col',
                                       gridspec_kw={'height_ratios': [12, 12, 1.6]})
    # ax.set_xlabel('Signal Energy')
    ax.set_title(f"\\(v = \\) {airspeeds[airspeed]} m/s"
                 if airspeed is not None
                 else "All airspeeds (\\(v\\)) considered")

    ax.set_ylabel('Angle of Attack')
    ax2.set_ylabel('P[stall | X=x]')
    ax3.set_ylabel('SE')
    if undertitle:
        ax3.set_xlabel(r'Signal Energy $(V^2 \cdot s)$')

    # Plot 1
    means_, stds_, _, _, stall_per_dist = data_003.dists_given_airspeed(airspeed, False)
    plot_003.plot(ax, means_, stds_, total, gamma, stall_per_dist)
    # ax.set_yticks([])

    # Plot 2
    ax2.plot(xs, np.vectorize(se.pdf_blue)(xs), color='blue')
    region2 = se.plot_certainty_region(ax2, xs, 0.09, '0.2', shift_y=0.13)

    # Plot 3
    region1_ = xs.reshape((-1, 1)) * np.ones((xs.size, means.size))
    means__ = means.reshape((1, -1))
    stds__ = stds.reshape((1, -1))
    region1_ = np.bitwise_and(means__-gamma*stds__ < region1_,
                              region1_ < means__+gamma*stds__)
    region1 = region1_[:, 0]
    for i in range(1, means.size):
        region1 = np.bitwise_or(region1, region1_[:, i])

    region = region1 * region2

    region_verts = [(xs[0], 0), *zip(xs, region), (xs[-1], 0)]
    region_poly = Polygon(region_verts, facecolor='#60af3d', edgecolor='#fff')
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
        plt.savefig(f'plots/{save_as}.pgf')
        plt.savefig(f'plots/{save_as}.png')
    else:
        plt.show()


if __name__ == '__main__':
    data_003.legacy_data(revert_to_bad=True)
    data.legacy_data(revert_to_bad=True)

    # fig_plot(gamma=4, confidence=0.999, airspeed=0)  # , subplot_adjust_params={'left': 0.05})

    # fig_plot(2, 0.9, 0, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0.15},
    #          save_as='plot_006_airspeed_6m_m2_tau.9', undertitle=False)
    # fig_plot(2, 0.9, 12, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0},
    #          save_as='plot_006_airspeed_20m_m2_tau.9')
    # fig_plot(2, 0.9, None, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0},
    #          right_limit=9000, save_as='plot_006_all_airspeeds_capped_m2_tau.9',
    #          undertitle=False)
    # fig_plot(4, 0.999, 0, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0.15},
    #          save_as='plot_006_airspeed_6m', undertitle=False)
    # fig_plot(4, 0.999, 12, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0},
    #          save_as='plot_006_airspeed_20m')
    # fig_plot(4, 0.999, None, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0},
    #          right_limit=9000, save_as='plot_006_all_airspeeds_capped', undertitle=False)

    # fig_plot(4, 0.999, None, subplot_adjust_params={}, save_as='plot_006_all_airspeeds')

    tau = .9
    z = 2
    # fig_plot(z, tau, 0, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0.15},
    #          save_as='animation-2/6m/plot_006_airspeed_6m_m2_tau.9', undertitle=False)
    fig_plot(z, tau, 0, subplot_adjust_params={'top': 0.95, 'bottom': 0.12, 'left': 0.15},
             undertitle=False)
