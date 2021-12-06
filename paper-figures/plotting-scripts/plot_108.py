from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import multivariate_normal
# import scipy.linalg as linalg
import matplotlib.colors as matcolors
import matplotlib.cm as cm

from enum import Enum

import data_104 as data_experiments
from misc import Plot

from typing import Any, List, Union, Tuple, Optional

dirname = "plots/plot_108"


class PlotType(Enum):
    Zpred = 0
    TauConf = 1
    SafetyEnv = 2


def compute_z_pred_gen_old(
    XY: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    z: float
) -> np.ndarray:
    assert len(XY.shape) == 2 and XY.shape[0] == 2
    # n = XY.shape[1]
    dists = means.shape[0]
    dim = means.shape[1]
    # Z = np.zeros((dists, n), dtype=float)

    mahalanobis = np.vectorize(
        distance.mahalanobis,
        signature='(n),(n),(n,n)->()',
        otypes=[float]
    )

    IV = np.linalg.inv(covs)  # .reshape((dists, 1, dim, dim))
    means = means.reshape((dists, 1, dim))
    # Z = vectorized_mahalonobis(XY.T, means, IV)
    Z = mahalanobis(XY.T, means, IV)
    # for j, (mean, var) in enumerate(zip(means, covs)):
    #     IV = np.linalg.inv(var)
    #     for i, xy in enumerate(XY.T):
    #         Z[j, i] = distance.mahalanobis(xy, mean, IV)

    # return Z.min(axis=0)  # type: ignore
    return Z.min(axis=0) < z  # type: ignore


# Vectorized (sped up version) of scipy.spatial.distance.mahalanobis
# It requires:
# * u to have the shape (data_n, dim)
# * v to have the shape (dists, dim)
# * IV to have the shape (dists, dim, dim)
# The result is of the shape (dists, data_n)
#
# The equation implemented is: \sqrt{ (u-v) V^{-1} (u-v)^T }
def vectorized_mahalonobis(u: np.ndarray, v: np.ndarray, IV: np.ndarray) -> np.ndarray:
    assert len(u.shape) == 2
    dim = u.shape[1]
    assert len(v.shape) == 2 and v.shape[1] == dim
    dists = v.shape[0]
    assert len(IV.shape) == 3 and IV.shape == (dists, dim, dim)

    v = v.reshape((dists, 1, dim))

    delta = u - v
    delta = delta.reshape((dists, -1, 1, dim))
    IV = IV.reshape((dists, 1, dim, dim))
    delta_T = delta.transpose((0, 1, 3, 2))

    m = (delta @ IV) @ delta_T
    m = m.reshape(m.shape[:2])  # the last two dimensions are 1
    return np.sqrt(m)  # type: ignore


def compute_stall_prob(
    xs: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    stall: np.ndarray
) -> np.ndarray:
    assert len(xs.shape) == 2
    n_data = xs.shape[0]
    # dim = xs.shape[1]

    ys_stall = np.zeros((n_data,))
    ys_nostall = np.zeros((n_data,))

    for mean, cov, stall_ in zip(means, covs, stall):
        mn = multivariate_normal(mean, cov)
        prob = mn.pdf(xs)
        if stall_:
            ys_stall += prob
        else:
            ys_nostall += prob

    # ys_stall = norm.pdf(xs.reshape((1, -1)),
    #                     loc=stall_means, scale=stall_stds)
    # ys_nostall = norm.pdf(xs.reshape((1, -1)),
    #                       loc=nostall_means, scale=nostall_stds)
    # red_prop = ys_stall.sum(axis=0)
    # blue_prop = ys_nostall.sum(axis=0)
    # return red_prop / (red_prop + blue_prop)  # type: ignore
    return ys_stall / (ys_stall + ys_nostall)  # type: ignore


def compute_z_pred_gen(
    XY: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    z: float
) -> np.ndarray:
    assert len(XY.shape) == 2 and XY.shape[0] == 2

    IV = np.linalg.inv(covs)
    # This is the simplest algorithm. It uses too much memory in some cases (with a large
    # number of distributions)
    # Z = vectorized_mahalonobis(XY.T, means, IV)
    # return Z.min(axis=0) < z  # type: ignore

    dim = covs.shape[1]
    Z = np.ones((1, XY.shape[1])) * float('inf')
    for mean, iv in zip(means, IV):
        Z_ = vectorized_mahalonobis(XY.T, mean.reshape((1, dim)), iv.reshape((1, dim, dim)))
        smaller = Z_ < Z
        Z[smaller] = Z_[smaller]
    return Z.flatten() < z  # type: ignore


# Modified from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    ax: Any,
    n_std: float = 3.0,
    facecolor: str = 'none',
    **kwargs: Any
) -> Any:
    assert len(mean.shape) == 1 and mean.shape == (2,)
    assert len(cov.shape) == 2 and cov.shape == (2, 2)

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        facecolor=facecolor, **kwargs)

    # calculating the stdandard deviation
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipses_model(
    means: np.ndarray,
    covs: np.ndarray,
    stall: np.ndarray,
    ax: Any,
    n_std: float = 3.0,
    colors: List[str] = ['blue', 'red']
) -> None:
    for mean, cov, stall_ in zip(means, covs, stall):
        color = colors[1 if stall_ else 0]
        confidence_ellipse(mean, cov, ax, n_std=n_std, edgecolor=color,
                           linestyle='--')


def plot_SE_2sensors(  # noqa: C901
    means: np.ndarray,
    covs: np.ndarray,
    stall: np.ndarray,
    z: float,
    tau: float,
    x_max: float,
    y_max: float,
    plot_density: int = 100,
    plot_type: PlotType = PlotType.SafetyEnv,
    show_ellipses: Union[bool, float] = True,
    no_ylabel: bool = False,
    no_xlabel: bool = False,
    no_title: bool = False,
    sensors: Tuple[int, int] = (0, 1)
) -> Any:
    X = np.linspace(0, x_max, plot_density)
    Y = np.linspace(0, y_max, plot_density)
    X, Y = np.meshgrid(X, Y)
    XY = np.array([X.flatten(), Y.flatten()])

    width = 3.7 if plot_type == PlotType.TauConf else 3
    fig, ax = plt.subplots(figsize=(width, 3))

    if plot_type == PlotType.Zpred:
        zpred = compute_z_pred_gen(XY, means, covs, z)
        zpred = zpred.reshape((plot_density, plot_density))
        ax.contourf(X, Y, zpred, 1, colors=['#FFFFFF', "#00d548"])
        if not no_title:
            ax.set_title("$z$-predictability")

    if plot_type == PlotType.TauConf:
        n_bin = 8
        cmap = matcolors.LinearSegmentedColormap.from_list(
            'cust_soft', ["#49f", "#f94"], N=n_bin)

        stall_prob = compute_stall_prob(XY.T, means, covs, stall)
        stall_prob = stall_prob.reshape((plot_density, plot_density))
        ax.contourf(X, Y, stall_prob, n_bin, cmap=cmap)

        ax.contour(X, Y, stall_prob < (1 - tau), 1, colors=['#26558d'])
        ax.contour(X, Y, stall_prob > tau, 1, colors=['#d66100'])

        norm = matcolors.Normalize(0, 1)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        # cmap = matcolors.LinearSegmentedColormap.from_list(
        #     'cust_line', ['#26558d', '#d66100'], N=2)
        # tau_conf = 1 * (stall_prob < (1 - tau)) + 2 * (stall_prob > tau)
        # ax.contour(X, Y, tau_conf, 1, cmap=cmap)
        if not no_title:
            ax.set_title(r"$\tau$-confidence")

    if plot_type == PlotType.SafetyEnv:
        zpred = compute_z_pred_gen(XY, means, covs, z)
        stall_prob = compute_stall_prob(XY.T, means, covs, stall)
        tau_conf = 1 * (stall_prob < (1 - tau)) + 2 * (stall_prob > tau)
        saf_env = zpred * tau_conf
        saf_env = saf_env.reshape((plot_density, plot_density))

        ax.contourf(X, Y, saf_env, 2, colors=['#ffffff', '#AAAAff', '#ffAAAA'])
        if not no_title:
            ax.set_title("Safety envelopes")

    if show_ellipses:
        if isinstance(show_ellipses, float):
            n_std = show_ellipses
        else:
            n_std = z
        ax.scatter(means[stall == 0, 0], means[stall == 0, 1], marker='.',
                   color="#26558d")
        ax.scatter(means[stall == 1, 0], means[stall == 1, 1], marker='.',
                   color="#d66100")
        confidence_ellipses_model(means, covs, stall, ax, n_std=n_std,
                                  colors=["#26558d", "#d66100"])
        ax.set_xlim(left=0.0, right=x_max)
        ax.set_ylim(bottom=0.0, top=y_max)

    # if False:
    #     if stall.size < 18:
    #         # Extending stall to encompass 18 values. The last values are -1
    #         stall = np.concatenate([stall, -1 * np.ones(18 - stall.size)])

    #     # Plotting raw data
    #     ax.scatter(data_sen1[:, stall == 0], data_sen2[:, stall == 0],
    #                marker='.', label="No-stall")
    #     ax.scatter(data_sen1[:, stall == 1], data_sen2[:, stall == 1],
    #                marker='x', label="Stall")

    if not no_xlabel:
        ax.set_xlabel(f"Sensor {sensors[0]+1} signal energy")
    if no_ylabel:
        ax.set_yticks([])
    else:
        ax.set_ylabel(f"Sensor {sensors[1]+1} signal energy")

    return fig, ax


def compute_metrics(
    total: int,
    means: np.ndarray,
    covs: np.ndarray,
    stall: np.ndarray,
    z: float,
    tau: float,
    n_sample: int = 10000,
    w: float = 10
) -> Tuple[float, float, float]:
    error = 0.0
    accuracy = 0.0

    n_sampled_total_neg = 0
    for (mean, cov, stall_) in zip(means, covs, stall):
        sample = multivariate_normal.rvs(mean=mean, cov=cov, size=n_sample)

        zpred = compute_z_pred_gen(sample.T, means, covs, z)
        stall_prob = compute_stall_prob(sample, means, covs, stall)
        tau_conf = 1 * (stall_prob < (1 - tau)) + 2 * (stall_prob > tau)
        classification = zpred * tau_conf - 1

        # ignoring negative points. They don't make sense for signal energies
        negative = sample.min(axis=1) < 0
        classification[negative] = -1
        n_sampled_total_neg += int(np.sum(negative))

        accuracy += np.sum(classification == stall_)
        error += np.sum(classification == (not stall_))

    accuracy /= total * n_sample - n_sampled_total_neg
    error /= total * n_sample - n_sampled_total_neg
    quailty = accuracy * (1 - error)**w
    return accuracy, error, quailty


if False and __name__ == '__main__':
    airspeed: Optional[int] = 10
    sensor1 = 0  # Numbering starting at zero
    sensor2 = 6
    z = 3
    tau = .99

    assert isinstance(airspeed, int)
    airspeed_i = data_experiments.airspeed_index(airspeed)
    assert isinstance(airspeed_i, int)
    total, means, covs, stall = \
        data_experiments.dists_given_airspeed(airspeed, (sensor1, sensor2))

    data_sen1 = data_experiments.seTW[:, sensor1, :, airspeed_i]
    data_sen2 = data_experiments.seTW[:, sensor2, :, airspeed_i]

    for plot_type in PlotType:
        with Plot(dirname,
                  f"{plot_type.name}-airspeed={airspeed}-sensors={sensor1+1}n{sensor2+1}"
                  f"-z={z}-tau={tau}"):
            plot_SE_2sensors(means, covs, stall, z, tau,
                             x_max=6, y_max=0.3, plot_density=300,
                             plot_type=plot_type, no_ylabel=plot_type != PlotType.Zpred,
                             sensors=(sensor1, sensor2), no_xlabel=True)

    accuracy, error, quailty = compute_metrics(total, means, covs, stall,
                                               z=z, tau=tau, n_sample=100000, w=10)

    print(f"accuracy = {accuracy*100:.3f}")
    print(f"error = {error*100:.3f}")
    print(f"quailty = {quailty*100:.3f}")

    # accuracy = 99.620
    # error = 0.011
    # quailty = 99.515


if False and __name__ == '__main__':
    airspeed = 20
    sensor1 = 0  # Numbering starting at zero
    sensor2 = 6
    z = 3
    tau = .99

    airspeed_i = data_experiments.airspeed_index(airspeed)
    assert isinstance(airspeed_i, int)
    total, means, covs, stall = \
        data_experiments.dists_given_airspeed(airspeed, (sensor1, sensor2))

    data_sen1 = data_experiments.seTW[:, sensor1, :, airspeed_i]
    data_sen2 = data_experiments.seTW[:, sensor2, :, airspeed_i]
    x_max = data_sen1.max()
    y_max = data_sen2.max()

    for plot_type in PlotType:
        with Plot(dirname,
                  f"{plot_type.name}-airspeed={airspeed}-sensors={sensor1+1}n{sensor2+1}"
                  f"-z={z}-tau={tau}"):
            plot_SE_2sensors(means, covs, stall, z, tau,
                             x_max=x_max, y_max=y_max, plot_density=300,
                             plot_type=plot_type, no_ylabel=plot_type != PlotType.Zpred,
                             sensors=(sensor1, sensor2),
                             no_title=True, no_xlabel=True)

    accuracy, error, quailty = compute_metrics(total, means, covs, stall,
                                               z=z, tau=tau, n_sample=100000, w=10)

    print(f"accuracy = {accuracy*100:.3f}")
    print(f"error = {error*100:.3f}")
    print(f"quailty = {quailty*100:.3f}")

    # accuracy = 91.790
    # error = 0.047
    # quailty = 91.361


if False and __name__ == '__main__':
    airspeed = None
    sensor1 = 0  # Numbering starting at zero
    sensor2 = 6
    z = 2
    tau = .9

    total, means, covs, stall = \
        data_experiments.dists_given_airspeed(airspeed, (sensor1, sensor2))

    # data_sen1 = data_experiments.seTW[:, sensor1, :, :]
    # data_sen2 = data_experiments.seTW[:, sensor2, :, :]
    # x_max = data_sen1.max()
    # y_max = data_sen2.max()

    for plot_type in PlotType:
        with Plot(dirname,
                  f"{plot_type.name}-all_airspeeds-sensors={sensor1+1}n{sensor2+1}"
                  f"-z={z}-tau={tau}"):
            plot_SE_2sensors(means, covs, stall, z, tau,
                             x_max=20, y_max=1.7, plot_density=300,
                             plot_type=plot_type, no_ylabel=plot_type != PlotType.Zpred,
                             sensors=(sensor1, sensor2),
                             show_ellipses=plot_type == PlotType.Zpred,
                             no_title=True)

    accuracy, error, quailty = compute_metrics(total, means, covs, stall,
                                               z=z, tau=tau, n_sample=10000, w=10)

    print(f"accuracy = {accuracy*100:.3f}")
    print(f"error = {error*100:.3f}")
    print(f"quailty = {quailty*100:.3f}")

    # FOR: tau = .99 and z = 3
    # accuracy = 55.581
    # error = 0.166
    # quailty = 54.663

    # FOR: tau = .90 and z = 2
    # accuracy = 78.959
    # error = 0.994
    # quailty = 71.449

    # FOR: tau = .90 and z = 3
    # accuracy = 79.060
    # error = 1.001
    # quailty = 71.497
