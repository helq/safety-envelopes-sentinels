from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

from misc import Plot

dirname = 'plots/plot_107'


def dquality(x_blue: np.ndarray,
             red_mean: float, w: float = 1
             ) -> np.ndarray:
    blue_mean = -red_mean
    pdf_blue = norm.pdf(x_blue, loc=blue_mean, scale=1.0)
    pdf_red = norm.pdf(x_blue, loc=red_mean, scale=1.0)
    cdf_blue = norm.cdf(x_blue, loc=blue_mean, scale=1.0)
    cdf_red = norm.cdf(x_blue, loc=red_mean, scale=1.0)
    return (pdf_blue * (1-cdf_red)**w  # type: ignore
            - cdf_blue * w * (1-cdf_red)**(w-1) * pdf_red)
    # return (w * pdf_red + (1 - (pdf_blue + pdf_red)))  # type: ignore


def best_quality_tau(w: float, red_mean: float, max_ptau: float = 4) -> float:
    blue_mean = -red_mean
    # This basically finds the point without doing the fsolve step.
    # I run fsolve anyway to get closer to the value
    pseudo_tau = np.linspace(0, max_ptau, 1000)  # to get a better estimate for max tau
    # pseudo_tau = np.linspace(0, 4)
    tau = norm.cdf(pseudo_tau)
    x_blue = 1/(2*red_mean) * (np.log(1-tau) - np.log(tau))
    accuracy = norm.cdf(x_blue, loc=blue_mean, scale=1.0)  # cdf_blue
    error = norm.cdf(x_blue, loc=red_mean, scale=1.0)  # cdf_red

    # Metric 1
    quality = accuracy * (1-error)**w
    start_i = quality.argmax()

    # Metric 2
    # missed = 1 - (error + accuracy)
    # quality = w * error + missed
    # start_i = quality.argmin()

    # Metric 3
    # coverage = accuracy + error
    # quality = coverage + w * (1-error)
    # start_i = quality.argmax()

    # Metric 4
    # coverage = accuracy + error
    # quality = coverage + (1-error) * np.exp(w)
    # start_i = quality.argmax()

    opt_tau = tau[start_i]  # this is very close to the optimal

    # return float(opt_tau)
    return float(
        fsolve(lambda tau:
               dquality(1 / (2*red_mean) * (np.log(1-tau) - np.log(tau)), red_mean, w=w),
               opt_tau))


if False and __name__ == '__main__':
    # for red_mean in [0.1, 0.5, 1, 1.5, 2, 3, 5, 10]:
    for red_mean in [0.1, 1, 3]:
        blue_mean = -red_mean

        ws = np.linspace(1, 100, 100)
        taus = np.array([best_quality_tau(w, red_mean) for w in ws])

        x_blue = 1/(2*red_mean) * (np.log(1-taus) - np.log(taus))
        accuracy = norm.cdf(x_blue, loc=blue_mean, scale=1.0)  # cdf_blue
        error = norm.cdf(x_blue, loc=red_mean, scale=1.0)  # cdf_red

        # with Plot():
        with Plot(f"{dirname}/optimal-tau",
                  f"optimal-tau-given-w-for-mean={red_mean:.1f}",
                  font_size=20):
            fig, ax = plt.subplots(figsize=(5, 3))
            # ax.plot(ws, norm.ppf(taus), label=r'optimal \(inv-\tau\)')
            ax.plot(ws, taus, color='orange', label=r'optimal \(\tau\)')
            ax.plot(ws, accuracy, color='green', label=r'\(accuracy\)')
            ax.plot(ws, error, color='red', label=r'\(error\)')
            ax.plot(ws, accuracy + error, color='blue', label=r'\(coverage\)')
            ax.set_title(f"\\(\\sigma={2*red_mean:.1f}\\)")
            if red_mean != 0.1:
                ax.set_yticks([])
            if red_mean == 1:
                ax.set_xlabel(r'\(w\)')
            if red_mean == 3:
                ax.legend()

if True and __name__ == '__main__':
    red_mean = np.linspace(0, 4)
    blue_mean = -red_mean

    # tau = 0.5
    # x_blue = 1/(2*red_mean) * (np.log(1-tau) - np.log(tau))  # = 0.0
    x_blue = 0.0
    accuracy = norm.cdf(x_blue, loc=blue_mean, scale=1.0)  # cdf_blue
    error = norm.cdf(x_blue, loc=red_mean, scale=1.0)  # cdf_red

    with Plot(f"{dirname}/metrics-given-sep", "tau=0.5", font_size=20):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(red_mean*2, accuracy, color='green', label=r'\(accuracy\)')
        ax.plot(red_mean*2, error, color='red', label=r'\(error\)')
        ax.plot(red_mean*2, accuracy + error, color='blue', label=r'\(coverage\)')
        ax.set_xlabel(r'\(\sigma\)')
        # ax.set_ylabel('optimum tau')
        ax.legend()


if False and __name__ == '__main__':
    std = 1
    red_mean = 0.1
    blue_mean = -red_mean

    fig, ax = plt.subplots()

    # xs = np.linspace(-6, 6, 100)
    # w = 10
    # ax.plot(xs, dquality(xs, w=w))
    # start = blue_mean-w/(120*red_mean)+0.5 + (0.1*red_mean)
    # ax.scatter(start, dquality(start, w))  # type: ignore
    # print(fsolve(lambda x: dquality(x, w=w), start))

    # def best_quality(w: float) -> float:
    #     if red_mean == 1:
    #         start = blue_mean-w/100+0.5
    #     if red_mean == 3:
    #         start = blue_mean-w/(120*red_mean)+0.8
    #     return float(fsolve(lambda x: dquality(x, w=w), start))

    # ws = np.linspace(1, 100, 100)
    # ax.plot(ws, [best_quality(w) for w in ws])
    # ax.set_xlabel('w')
    # ax.set_ylabel('optimum x')

    # def x_eq_tau(tau: np.ndarray, x: float) -> np.ndarray:
    #     return x - 1 / (2*red_mean) * (np.log(1-tau) - np.log(tau))  # type: ignore

    # def best_quality_tau(w: float) -> float:
    #     x = best_quality(w)
    #     return float(fsolve(lambda t: x_eq_tau(t, x), 0.97))

    # taus = np.linspace(0.00000001, 0.99999999, 100)
    # ax.plot(taus, x_eq_tau(taus, x=-0.6807031871140989))
    # ax.set_xlabel('tau')
    # print(best_quality_tau(w=60))

    ws = np.linspace(1, 100, 100)
    taus = np.array([best_quality_tau(w) for w in ws])

    x_blue = 1/(2*red_mean) * (np.log(1-taus) - np.log(taus))
    accuracy = norm.cdf(x_blue, loc=blue_mean, scale=std)  # cdf_blue
    error = norm.cdf(x_blue, loc=red_mean, scale=std)  # cdf_red

    ax.plot(ws, taus, label=r'optimal \(\tau\)')
    ax.plot(ws, accuracy, label=r'\(accuracy\)')
    ax.plot(ws, error, label=r'\(error\)')
    ax.set_xlabel('w')
    # ax.set_ylabel('optimum tau')
    ax.legend()
    plt.show()

if False and __name__ == '__main__':
    std = 1
    red_mean = 1
    blue_mean = -red_mean

    w = 10
    pseudo_tau = np.linspace(0, 4)
    tau = norm.cdf(pseudo_tau)
    x_blue = 1/(2*red_mean) * (np.log(1-tau) - np.log(tau))
    accuracy = norm.cdf(x_blue, loc=blue_mean, scale=std)  # cdf_blue
    error = norm.cdf(x_blue, loc=red_mean, scale=std)  # cdf_red

    fig, ax = plt.subplots()
    ax.plot(pseudo_tau, accuracy, label='accuracy')
    ax.plot(pseudo_tau, error, label='error')
    ax.plot(pseudo_tau, accuracy * (1-error)**w, label=f'accuracy * (1-error)**{w}')
    # ax.plot(pseudo_tau, 2*norm.pdf(pseudo_tau, scale=1.2), label='normal dist')
    # ax.plot(pseudo_tau, accuracy * (1-w*error), label=f'accuracy * (1-{w}*error)')
    ax.set_xlabel("tau")
    ax.set_ylabel("metric")
    # ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '0.9999..' if x > 3.6 else f"{norm.cdf(x):.4f}"))
    ax.legend()
    plt.show()
