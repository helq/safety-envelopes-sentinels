from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import probability_001 as prob

# import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

if __name__ == '__main__':
    mat_contents = sio.loadmat('windTunnel_data_sensor3_AS15.mat')

    seTW = mat_contents['seTW']

    # computed total and stall dists, per airspeed
    total_all = [18, 18, 18, 18, 18, 18, 18, 18, 17, 16, 14, 13, 12, 9, 7]
    n_stall_all = [7, 7, 5, 5, 5, 5, 4, 4, 4, 4, 2, 2, 1, 0, 0]

    airspeed: Optional[int] = 5

    if airspeed is not None:
        total = total_all[airspeed]
        means = np.sum(seTW[:, 3, :total, airspeed], axis=0)
        stds = np.sqrt(
            np.sum((seTW[:, 3, :total, airspeed] - means.reshape((1, total)))**2, axis=0)
        ) / 90
        # total = 18  # means.size
        # n_stall = 5
        n_stall = n_stall_all[airspeed]

    else:  # all airspeeds at the same time
        means_red = []
        stds_red = []
        means_blue = []
        stds_blue = []
        for airspeed, (t, n_s) in enumerate(zip(total_all, n_stall_all)):
            means = np.sum(seTW[:, 3, :t, airspeed], axis=0)
            stds = (np.sqrt(np.sum((seTW[:, 3, :t, airspeed] - means.reshape((1, t)))**2, axis=0))
                    / 90)
            means_red.append(means[:-n_s])
            means_blue.append(means[-n_s:])
            stds_red.append(stds[:-n_s])
            stds_blue.append(stds[-n_s:])

        means_red_ = np.concatenate(means_red)
        stds_red_ = np.concatenate(stds_red)
        means_blue_ = np.concatenate(means_blue)
        stds_blue_ = np.concatenate(stds_blue)

        n_stall = stds_blue_.size
        means = np.concatenate([means_red_, means_blue_])
        stds = np.concatenate([stds_red_, stds_blue_])
        total = means.size

    # exit(1)
    print("Means")
    print(means)
    print("Standard deviations")
    print(stds)
    print("Total number of distributions:", total)
    print("Number of distributions in stall:", n_stall)

    # means = mat_contents['meanSE'][0, :]
    # stds = mat_contents['stdSE'][0, :]
    # total = 18
    # n_stall = 4
    means_1, stds_1 = means[:-n_stall], stds[:-n_stall]
    means_2, stds_2 = means[-n_stall:], stds[-n_stall:]

    # means_1, stds_1 = np.array([4]), np.array([1])
    # means_2, stds_2 = np.array([8.4]), np.array([1.2])
    # means, stds = np.array([4, 8.4]), np.array([1, 1.2])
    # total = 2
    # n_stall = 1

    # means_1, stds_1 = np.array([1, 4, 9, 3.5, 2, 5]), np.array([.3, 1, 3, .3, .5, 1.5])
    # means_2, stds_2 = np.array([9, 11, 14]), np.array([2, 1.6, 2.2])

    # means_1, stds_1 = ex.means_red[0], ex.stds_red[0]
    # means_2, stds_2 = ex.means_blue[0], ex.stds_blue[0]

    gamma_to_plot = 3.5
    # left_plot = min(means_1[0] - gamma_to_plot*stds_1[0],
    #                 means_2[0] - gamma_to_plot*stds_2[0])
    right_plot = max(means_1[-1] + gamma_to_plot*stds_1[-1],
                     means_2[-1] + gamma_to_plot*stds_2[-1])
    left_plot = 0
    # right_plot = 13
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

    se = prob.wing_BIWT(means, stds, total-n_stall, 0.95, 0)
    # Enumerating probability of no-stall given a distribution
    # print([(i, rgd.p) for i, rgd in enumerate(se.red_given_dist)])

    # ## Plotting code
    ##################################
    fig, ax = plt.subplots()
    # ax.set_yscale('log')
    plt.xlabel('Signal Energy')
    plt.ylabel('pdf(x)')

    # se.plot(ax, xs)
    # se.plot(ax, xs, equal_size=False)
    height = se.plot(ax, xs, region=False)
    xs = np.arange(left_plot, right_plot, (right_plot-left_plot)/10000)
    se.plot_certainty_region(ax, xs, height, '0.2')
    # se2.plot(ax, xs)
    # se.plot(ax, xs, False)

    # plt.savefig('plots/plot_001.pgf')
    plt.show()
