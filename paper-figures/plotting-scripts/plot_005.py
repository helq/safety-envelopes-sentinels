from __future__ import annotations

from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import probability_005 as prob
import data_005 as data

import matplotlib


def fig_plot(
    gamma: int,
    taus: List[Tuple[int, float]],
    airspeed: Optional[int],
    right_limit: Optional[float] = None,
    save_as: Optional[str] = None,
    subplot_adjust_params: Optional[Dict[str, float]] = None,
    text: Optional[Tuple[float, float, str]] = None,
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

    # Plotting parameters
    i = np.where(means == means.max())[0].flatten()[0]
    right_plot = int(right_limit if right_limit is not None else means[i] + gamma*stds[i])
    left_plot = 0
    plot_density = 1000

    ##################################
    # Plotting prep
    # xs = np.arange(left_plot, right_plot, 1)
    xs = np.arange(left_plot, right_plot, (right_plot-left_plot)/plot_density)

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

    tau = 0.9999  # Dummy value, will be overwritten
    se = prob.wing_BIWT(means, stds, total-n_stall, tau, 0)

    xs = np.arange(left_plot, right_plot, (right_plot-left_plot)/10000)
    ys = np.vectorize(se.pdf_blue)(xs)

    # ## Plotting code
    ##################################

    for i, tau in taus:
        fig, ax = plt.subplots()
        # ax.set_yscale('log')
        plt.xlabel('Signal Energy')
        plt.ylabel(r'\(P[stall \mid X=x]\)')

        ax.plot(xs, ys, color='blue')
        se.confidence = tau
        se.plot_certainty_region(ax, xs, 0.09, '0.2', shift_y=0.13)

        if subplot_adjust_params is not None:
            params = {'top': 0.97, 'bottom': 0.1, 'right': 0.97,
                      'left': 0.08, 'hspace': 0, 'wspace': 0}
            params.update(subplot_adjust_params)
            plt.subplots_adjust(**params)

        if text:
            ax.text(text[0], text[1], text[2].format(i=i, tau=tau))

        if right_limit is not None:
            ax.set_xlim(right=right_limit)

        plt.savefig(f'plots/{save_as}.png'.format(i=i, tau=tau))


if __name__ == '__main__':
    data.legacy_data(revert_to_bad=True)

    # fig_plot(gamma=4, confidence=0.999, airspeed=12)

    # fig_plot(4, 0.999, 0, save_as='animation-2/plot_005_airspeed_6m',
    #          subplot_adjust_params={'left': 0.15, 'bottom': 0.18})
    # fig_plot(4, 0.999, 12, save_as='plot_005_airspeed_20m')
    # fig_plot(4, 0.999, None, save_as='plot_005_all_airspeeds')
    # fig_plot(4, 0.999, None, right_limit=9000, save_as='plot_005_all_airspeeds_capped')

    subplot_params = {'left': 0.15, 'bottom': 0.18}
    # taus = [(-2, 0.5003), (-1, .9999)]
    taus = list(enumerate(np.log(np.arange(0, 1, 0.01) + 1)/(2*np.log(2))+.503))
    fig_plot(4, taus, 0,
             right_limit=800,
             save_as='animation-2/6ms/{i:03d}-plot_005_airspeed_6m-tau={tau:.4f}',
             subplot_adjust_params=subplot_params)

    fig_plot(4, taus, 12,
             right_limit=11000,
             save_as='animation-2/20ms/{i:03d}-plot_005_airspeed_20m-tau={tau:.4f}',
             subplot_adjust_params=subplot_params,
             text=(6000, 0.15, "\\(\\tau\\) = {tau:.4f}"))

    # fig_plot(4, taus, None,
    #          right_limit=9000,
    #          save_as='animation-2/all/{i:03d}-plot_005_all_airspeeds-tau={tau:.4f}',
    #          subplot_adjust_params=subplot_params)
