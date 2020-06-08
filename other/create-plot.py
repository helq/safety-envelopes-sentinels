from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ast

from typing import Optional, List, Any


def readtxt(file: str) -> List[Any]:
    return [ast.literal_eval(line) for line in open(file, 'r').readlines()]


def fig_plot(
    signal: np.array,
    mean_cf: np.array,
    sample_cf: np.array,
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

    x = np.linspace(1.0, len(signal), len(signal))
    x_sample = x[x.size-sample_cf.shape[0]:]

    fig, ax = plt.subplots(
        nrows=4, sharex='col', gridspec_kw={'height_ratios': [5, 2, 5, 2]},
        frameon=False
    )
    # ax[0].set_title('A tale of 2 subplots')
    ax[-1].set_xlabel('Measurement in time (s)')

    ax[0].plot(x, signal, '.-')
    ax[0].set_ylabel('Signal')

    color_mean_cf = [('#c85c46' if m == 0 else '#4677c8') for m in mean_cf[:, 1]]
    ax[1].scatter(x, mean_cf[:, 1], color=color_mean_cf, marker='.')
    ax[1].set_ylabel('Mean-\nconsistency')
    ax[1].set_ylim(bottom=-0.2, top=1.2)
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(['False', 'True'])

    # ax[2].plot(x_sample, np.sqrt(sample_cf[:, 1]), '.-')
    ax[2].errorbar(
        x_sample, sample_cf[:, 0],
        yerr=np.sqrt(sample_cf[:, 1]), ecolor='#c89388')
    ax[2].set_ylabel('Sample mean\nand stds')

    color_sample_cf = [('#c85c46' if m == 0 else '#4677c8') for m in sample_cf[:, 2]]
    ax[3].scatter(
        x_sample, sample_cf[:, 2],
        color=color_sample_cf, marker='.')
    ax[3].set_ylabel('Sample-\nconsistency')
    ax[3].set_ylim(bottom=-0.2, top=1.2)
    ax[3].set_yticks([0, 1])
    ax[3].set_yticklabels(['False', 'True'])

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.13, hspace=0, wspace=0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if save_as:
        plt.savefig(f'plots/{save_as}.pgf')
        plt.savefig(f'plots/{save_as}.png')
    else:
        plt.show()


if __name__ == '__main__':
    signal = np.array(readtxt('./generated.txt'))
    mean_cf = np.array(readtxt('./mean-consistency-check.txt'))
    sample_cf = np.array([(m, v, b) for (m, (v, b)) in readtxt('./sample-consistency-check.txt')])

    # TODO: PLOT USING ONE STANDARD DEVIATION (m should be 4)
    fig_plot(signal, mean_cf, sample_cf, save_as='consistency-m_mu_4-m_sigma_4')
    # fig_plot(signal, mean_cf, sample_cf)
