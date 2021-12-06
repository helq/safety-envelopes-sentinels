from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm
# from scipy.optimize import fsolve

from misc import Plot, horizontal_interval

dirname = 'plots/plot_105'


if __name__ == '__main__':
    xs = np.linspace(-4, 7, 250)
    mean_blue, std_blue = blue = 0, 1
    mean_red, std_red = red = 3, 1.4
    # blue_interval = (-4, 0.63)
    # red_interval = (1.897, 7)
    blue_interval = (-1.6, 1)
    red_interval = (1.5, 4.6)
    # blue_interval = (-4, 1.4043)
    # red_interval = (1.4043, 7)

    ys_red = norm.pdf(xs, loc=mean_red, scale=std_red)
    ys_blue = norm.pdf(xs, loc=mean_blue, scale=std_blue)

    # finding midpoint
    # midpoint = fsolve(lambda x:
    #                   norm.pdf(x, loc=mean_red, scale=std_red)
    #                   - norm.pdf(x, loc=mean_blue, scale=std_blue), 2)
    # blue_interval = (-4, midpoint[0])
    # red_interval = (midpoint[0], 7)

    with Plot(dirname, f"metrics_params_blue={blue}_red={red}"
              f"_intervals={blue_interval}-{red_interval}"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(xs, ys_red, color='red')
        ax.plot(xs, ys_blue, color='blue')

        # Accuracy
        red_accuracy = ys_red.copy()
        red_accuracy[xs < red_interval[0]] = 0
        red_accuracy[red_interval[1] < xs] = 0
        blue_accuracy = ys_blue.copy()
        blue_accuracy[xs < blue_interval[0]] = 0
        blue_accuracy[blue_interval[1] < xs] = 0

        # Error
        blue_error = ys_blue.copy()
        blue_error[xs < red_interval[0]] = 0
        blue_error[red_interval[1] < xs] = 0
        red_error = ys_red.copy()
        red_error[xs < blue_interval[0]] = 0
        red_error[blue_interval[1] < xs] = 0

        ax.fill_between(xs, red_accuracy, alpha=.3, color='red', label='accuracy red')
        ax.fill_between(xs, blue_accuracy, alpha=.3, color='blue', label='accuracy blue')
        ax.fill_between(xs, red_error, alpha=.3, color='red', hatch='////', label='error red')
        ax.fill_between(xs, blue_error, alpha=.3, color='blue', hatch='////', label='error blue')

        ax.add_collection(
            horizontal_interval([red_interval, blue_interval], colors=['red', 'blue']))
        ax.set_ylim(bottom=-.03)

        accuracy_patch = mpatches.Patch(color='black', alpha=.4, label='accuracy')
        error_patch = mpatches.Patch(color='black', alpha=.4, hatch='////', label='error')
        ax.legend(handles=[accuracy_patch, error_patch])
        # ax.legend()

        accuracy = (norm.cdf(blue_interval[1], loc=mean_blue, scale=std_blue) -
                    norm.cdf(blue_interval[0], loc=mean_blue, scale=std_blue)) + \
                   (norm.cdf(red_interval[1], loc=mean_red, scale=std_red) -
                    norm.cdf(red_interval[0], loc=mean_red, scale=std_red))
        accuracy /= 2
        error = (norm.cdf(red_interval[1], loc=mean_blue, scale=std_blue) -
                 norm.cdf(red_interval[0], loc=mean_blue, scale=std_blue)) + \
                (norm.cdf(blue_interval[1], loc=mean_red, scale=std_red) -
                 norm.cdf(blue_interval[0], loc=mean_red, scale=std_red))
        error /= 2
        coverage = accuracy + error

        ax.set_title(f"{accuracy*100:.2f}% accuracy - {error*100:.2f}% error - "
                     f"{coverage*100:.2f}% coverage")
