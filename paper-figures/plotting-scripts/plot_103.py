from __future__ import annotations

from typing import Optional, Iterator, Tuple, Dict, Any, Type, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
import matplotlib
# import multiprocessing
import pickle
import os.path
from abc import ABC, abstractmethod
from enum import Enum

import data_104 as data_experiments
import data_101 as data_artificial
import probability_101 as prob
from misc import region_from_positives


airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


class SEMetric(ABC):
    def __init__(self, store_file: str, param: Any, override: bool = False):
        self.store_file = store_file
        self.override = override
        self.param = param
        self.store: Dict[int, Dict[Tuple[float, float], float]]

        if os.path.isfile(store_file):
            with open(store_file, 'rb') as f:
                self.store = pickle.load(f)
        else:
            self.store = {}

        if param not in self.store:
            self.store[param] = {}

    def __del__(self) -> None:
        # print("my store =", self.store)
        with open(self.store_file, 'wb') as f:
            pickle.dump(self.store, f)

    @abstractmethod
    def call(self, val: Any) -> float:
        pass

    def __call__(self, val: Any) -> float:
        store_for_param = self.store[self.param]
        if val in store_for_param and not self.override:
            metric_res = store_for_param[val]
        else:
            metric_res = self.call(val)
            store_for_param[val] = metric_res

        return metric_res


class SESize(SEMetric):
    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        airspeed = -1 if airspeed is None else airspeed
        super().__init__("plots/plot_103/bad-metric/computation.data", airspeed, override)
        self.airspeed = airspeed
        self.means = means
        self.stds = stds
        self.stall = stall
        if not kargs:
            kargs = {'resolution_density': 1000}
        self.kargs = kargs

    def call(self, ztau: Tuple[float, float]) -> float:
        means = self.means
        stds = self.stds
        stall = self.stall

        return get_safety_envelope_size(means, stds, stall, ztau[0], ztau[1], **self.kargs)


class MetricMode(Enum):
    Coverage = 1
    Accuracy = 2
    Error = 3


class PercentageOfPointsInSE(SEMetric):
    # Make sure to initialize this value!!
    signal_energies: Optional[np.ndarray] = None
    mode: MetricMode = MetricMode.Coverage

    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        assert self.signal_energies is not None, \
            "PercentageOfPointsInSE.signal_energies must be initialized before running the metric"

        mode = self.mode
        super().__init__("plots/plot_103/pauls-metric/computation.data",
                         airspeed if mode is MetricMode.Coverage else (airspeed, mode),
                         override)
        self.airspeed = airspeed
        self.override = override

        self.se = prob.wing_SE_classification(means, stds, stall, 0.0)

    def call(self, ztau: Tuple[float, float]) -> float:
        z, tau = ztau
        self.se.confidence = tau

        assert self.signal_energies is not None
        N = len(self.signal_energies)

        # Checking whether the point lies inside or not of safety envelope
        inside = 0
        for x in self.signal_energies:
            if self.mode is MetricMode.Coverage:
                inside += self.se.inside_safety_envelope(x, z)
            elif self.mode is MetricMode.Accuracy:
                raise NotImplementedError("Sorry :S")
                if self.se.classify(x) == prob.TernaryClasses.Red:
                    inside += self.se.inside_safety_envelope(x, z)
            else:
                raise NotImplementedError("Sorry :S")
                inside += self.se.inside_safety_envelope(x, z)

        return float(inside)/N


# TODO: There is something weird going on with coverage. It is producing values bigger
# than 1. It shouldn't
class Coverage(SEMetric):
    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        airspeed = -1 if airspeed is None else airspeed
        super().__init__("plots/plot_103/coverage/computation.data", airspeed, override)
        self.airspeed = airspeed
        self.means = means
        self.stds = stds
        self.stall = stall
        if not kargs:
            kargs = {'resolution_density': 1000}
        self.kargs = kargs

    def call(self, ztau: Tuple[float, float]) -> float:
        means = self.means
        stds = self.stds
        stall = self.stall

        return get_global_coverage(means, stds, stall, ztau[0], ztau[1], **self.kargs)


class Accuracy(SEMetric):
    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        airspeed = -1 if airspeed is None else airspeed
        super().__init__("plots/plot_103/accuracy/computation.data", airspeed, override)
        self.airspeed = airspeed
        self.means = means
        self.stds = stds
        self.stall = stall
        if not kargs:
            kargs = {'resolution_density': 1000}
        self.kargs = kargs

    def call(self, ztau: Tuple[float, float]) -> float:
        means = self.means
        stds = self.stds
        stall = self.stall

        return get_global_accuracy(means, stds, stall, ztau[0], ztau[1], **self.kargs)


class Error(SEMetric):
    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        airspeed = -1 if airspeed is None else airspeed
        super().__init__("plots/plot_103/error/computation.data", airspeed, override)
        self.airspeed = airspeed
        self.means = means
        self.stds = stds
        self.stall = stall
        if not kargs:
            kargs = {'resolution_density': 1000}
        self.kargs = kargs

    def call(self, ztau: Tuple[float, float]) -> float:
        means = self.means
        stds = self.stds
        stall = self.stall

        return get_global_accuracy(means, stds, stall, ztau[0], ztau[1], inverse=True, **self.kargs)


class Quality(SEMetric):
    weight: float = 1

    def __init__(self, airspeed: Optional[int],
                 means: np.ndarray, stds: np.ndarray, stall: np.ndarray,
                 override: bool, **kargs: Any):
        # override only is used to recompute the value with the already computed value
        # from accuracy and error. If you wish to recompute the values for accuracy and
        # error, you have to run those processes again with override true
        airspeed_ = -1 if airspeed is None else airspeed
        super().__init__("plots/plot_103/comb-accu-error/computation.data",
                         airspeed_ if self.weight == 1 else (airspeed_, self.weight),
                         override)
        self.accuracy_metric = Accuracy(airspeed, means, stds, stall, override=False, **kargs)
        self.error_metric = Error(airspeed, means, stds, stall, override=False, **kargs)

    def call(self, ztau: Tuple[float, float]) -> float:
        total_accuracy = self.accuracy_metric(ztau)
        total_error = self.error_metric(ztau)

        # return total_accuracy * (1 - total_error)
        # return total_accuracy * (1 - self.weight*total_error)
        return total_accuracy * (1 - total_error)**self.weight


def print_SE_size(
    airspeed: Optional[int],
    zs_n: int,
    taus_n: int,
    original_data: bool = False,
    skip: int = 0,
    is3d: bool = True,
    metric_init: Type[SEMetric] = SESize,
    save_as: Optional[str] = None,
    elev_azim: Optional[Tuple[float, float]] = None,
    override: bool = False,
    resolution_density: int = 1000
) -> None:
    if save_as:
        matplotlib.use("pgf")
        # matplotlib.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        #     'font.size': 30
        # })
    else:
        matplotlib.use('Qt5Agg')

    if original_data:
        _, means, vars_, stall = data_experiments.dists_given_airspeed(airspeed)
    else:
        assert isinstance(airspeed, int)
        _, means, vars_, stall = data_artificial.dists_given_airspeed(airspeed, skip=skip)
    stds = np.sqrt(vars_)

    zs_ = np.linspace(0.1, 4, zs_n)
    log_taus = np.linspace(1.1, 12, taus_n)

    taus = 1 - 1/2**log_taus
    print(f"taus: {taus}")
    print(f"zs: {zs_}")

    _, log_taus = np.meshgrid(zs_, log_taus)
    zs, taus = np.meshgrid(zs_, taus)

    metric_results = []
    # pool = multiprocessing.Pool(4)
    # metric_results = pool.map(SESize(airspeed, means, stds, stall),
    #                     zip(zs.flatten(), taus.flatten()))
    print("Computing safety envelopes for a grid of tau and z values")
    # n = zs.size
    biggest = (0.0, 0.0, 0.0)
    i = 0
    compute_metric = metric_init(airspeed, means, stds, stall, override,  # type: ignore
                                 resolution_density=resolution_density)
    for z, tau in zip(zs.flatten(), taus.flatten()):
        metric_res = compute_metric((z, tau))
        metric_results.append(metric_res)
        if i % 10 == 0:
            print(".", flush=True, end="")
        i += 1
        if biggest[2] < metric_res:
            biggest = (z, tau, metric_res)
        # print(f"metric res: {metric_res}")
    print()

    print(f"With params z={biggest[0]} and tau={biggest[1]}, the biggest value was = {biggest[2]}")

    se_sizes_ = np.array(metric_results).reshape(zs.shape)
    # print(se_sizes_)

    if is3d:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(zs, log_taus, se_sizes_, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # ax.yaxis.set_scale('log')
        ax.set_xlabel("z")
        ax.set_ylabel("tau")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: '0.999..' if x > 10 else f"{1 - 1/2**x:.3f}"))

        if elev_azim is not None:
            ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    else:
        plt.imshow(np.array(metric_results).reshape((taus_n, zs_n)),  # interpolation='nearest',
                   extent=(zs_.min(), zs_.max(), taus.min(), taus.max()),
                   cmap=cm.summer, aspect='auto')
        #            cmap=cm.RdBu)
        plt.yscale('log')

    if save_as:
        plt.savefig(f'plots/{save_as}.pgf', bbox_inches='tight')
        plt.savefig(f'plots/{save_as}.png', bbox_inches='tight')
    else:
        plt.show()


def print_SE_size_1D(  # noqa: C901
    airspeed: Optional[int],
    z: Optional[float] = None,
    tau: Optional[float] = None,
    original_data: bool = False,
    skip: int = 0,
    save_as: Optional[str] = None,
    elev_azim: Optional[Tuple[float, float]] = None,
    override: bool = False,
    resolution_density: int = 1000
) -> None:
    assert (z is None) != (tau is None), "Must pass either `z' or `tau' to function"

    if save_as:
        matplotlib.use("pgf")
    else:
        matplotlib.use('Qt5Agg')

    if original_data:
        _, means, vars_, stall = data_experiments.dists_given_airspeed(airspeed)
    else:
        assert isinstance(airspeed, int)
        _, means, vars_, stall = data_artificial.dists_given_airspeed(airspeed, skip=skip)
    stds = np.sqrt(vars_)

    if tau:
        # zs = np.linspace(0.1, 4, 40)
        zs = np.linspace(0.1, 4, 200)
        X = zs
    if z:
        # log_taus = np.linspace(1.1, 12, 40)
        log_taus = np.linspace(1.1, 15, 200)
        taus = 1 - 1/2**log_taus
        X = taus

    metric_results = []
    print("Computing safety envelopes for a grid of tau and z values")
    compute_se_size = SESize(airspeed, means, stds, stall, override,
                             resolution_density=resolution_density)
    for x in X:
        if tau:
            se_size = compute_se_size((x, tau))
        if z:
            se_size = compute_se_size((z, x))
        metric_results.append(se_size)
        print(".", flush=True, end="")
    print()

    se_sizes_np = np.array(metric_results).reshape(X.shape)

    fig, ax = plt.subplots()
    ax.plot(X, se_sizes_np)
    if tau:
        ax.set_title(f"Change of the safety envelope for tau = {tau:.4f} with varying z\n"
                     f"airspeed = {'ALL' if airspeed is None else airspeed}")
        ax.set_xlabel("z")
    if z:
        ax.set_title(f"Change of the safety envelope for tau = {z:.2f} with varying tau\n"
                     f"airspeed = {'ALL' if airspeed is None else airspeed}")
        ax.set_xscale('logit')
        ax.set_xlabel("tau")
    ax.set_ylabel("Safety Envelope size")

    if save_as:
        plt.savefig(f'plots/{save_as}.pgf')
        plt.savefig(f'plots/{save_as}.png')
    else:
        plt.show()


def get_safety_envelope_size(
    means: np.ndarray,
    stds: np.ndarray,
    stall: np.ndarray,
    z: float,
    tau: float,
    resolution_density: int = 1000
) -> float:
    se = prob.wing_SE_classification(means, stds, stall, tau)
    region_verts = get_region_verts(se, means, stds, z, resolution_density)
    return extract_safety_envelope_size(region_verts)


def get_global_coverage(
    means: np.ndarray,
    stds: np.ndarray,
    stall: np.ndarray,
    z: float,
    tau: float,
    resolution_density: int = 1000
) -> float:
    se = prob.wing_SE_classification(means, stds, stall, tau)
    region_verts = get_region_verts(se, means, stds, z, resolution_density)
    intervals = extract_intervals(region_verts)
    total = 0.0
    for interval in intervals:
        total += se.area_under_the_curve(interval)
    return total


def get_global_accuracy(
    means: np.ndarray,
    stds: np.ndarray,
    stall: np.ndarray,
    z: float,
    tau: float,
    inverse: bool = False,
    resolution_density: int = 1000
) -> float:
    se = prob.wing_SE_classification(means, stds, stall, tau)
    region_stall_verts, region_no_stall_verts = \
        get_stall_no_stall_region_verts(se, means, stds, z, resolution_density)
    total = 0.0
    intervals = extract_intervals(region_stall_verts)
    for interval in intervals:
        is_stall = False if inverse else True
        total += se.area_under_the_curve(interval, red_conditioned=is_stall)
    intervals = extract_intervals(region_no_stall_verts)
    for interval in intervals:
        is_no_stall = True if inverse else False
        total += se.area_under_the_curve(interval, red_conditioned=is_no_stall)
    return total


def get_pred_region(
    xs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    z: float
) -> np.ndarray:
    # Predictability region
    region1_ = xs.reshape((-1, 1)) * np.ones((xs.size, means.size))
    means__ = means.reshape((1, -1))
    stds__ = stds.reshape((1, -1))
    region1_ = np.bitwise_and(means__-z*stds__ < region1_,
                              region1_ < means__+z*stds__)
    region1 = region1_[:, 0]
    for i in range(1, means.size):
        region1 = np.bitwise_or(region1, region1_[:, i])

    return region1  # type: ignore


def get_region_verts(
    se: prob.SafetyEnvelopesClassification,
    means: np.ndarray,
    stds: np.ndarray,
    z: float,
    resolution_density: int = 1000
) -> Iterator[Tuple[float, float]]:
    left_plot = (means - z*stds).min()
    right_plot = (means + z*stds).max()

    xs = np.linspace(left_plot, right_plot, resolution_density)

    # Predictability region
    region1 = get_pred_region(xs, means, stds, z)

    # Certainty region
    region2 = se.plot_certainty_region(None, xs, 0.09, '0.2', shift_y=0.13)
    assert isinstance(region2, np.ndarray)

    # Safety Envelopes region
    region = region1 * region2

    return region_from_positives(zip(xs, region))


def get_stall_no_stall_region_verts(
    se: prob.SafetyEnvelopesClassification,
    means: np.ndarray,
    stds: np.ndarray,
    z: float,
    resolution_density: int = 1000
) -> Tuple[Iterator[Tuple[float, float]], Iterator[Tuple[float, float]]]:
    left_plot = (means - z*stds).min()
    right_plot = (means + z*stds).max()

    xs = np.linspace(left_plot, right_plot, resolution_density)

    # Predictability region
    region_pred = get_pred_region(xs, means, stds, z)

    # Certainty region
    region_cert = \
        se.plot_certainty_region(None, xs, 0.09, '0.2', shift_y=0.13, get_red_n_blue=True)
    assert isinstance(region_cert, tuple)
    region_stall, region_no_stall = region_cert

    # Safety Envelopes regions (separated in stall and no-stall)
    region1 = region_pred * region_stall
    region2 = region_pred * region_no_stall

    return region_from_positives(zip(xs, region1)), region_from_positives(zip(xs, region2))


def extract_safety_envelope_size(corners: Iterator[Tuple[float, float]]) -> float:
    total = 0.0
    prev_corn: Optional[float] = None
    for corn, val in corners:
        if val == 1.0:
            if prev_corn is None:
                prev_corn = corn
            else:
                total += corn - prev_corn
                prev_corn = None
    return total


def extract_intervals(corners: Iterator[Tuple[float, float]]
                      ) -> Iterator[Tuple[float, float]]:
    prev_corn: Optional[float] = None
    for corn, val in corners:
        if val == 1.0:
            if prev_corn is None:
                prev_corn = corn
            else:
                yield prev_corn, corn
                prev_corn = None


if False and __name__ == '__main__':
    airspeed = None
    print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True)
    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               save_as=f'plot_103/bad-metric/plot103-3d-airspeed-{airspeed}-default-view')
    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               elev_azim=(90, 270),
    #               save_as=f'plot_103/bad-metric/plot103-3d-airspeed-{airspeed}-on-top-view')

    tau = 1 - 1/2**12  # Very close to 100%
    z = 2
    print_SE_size_1D(airspeed, tau=tau, original_data=True)
    # print_SE_size_1D(airspeed, tau=tau, original_data=True,
    #                  save_as=f"plot_103/bad-metric/plot103-2d-airspeed-{airspeed}-tau={tau}")
    print_SE_size_1D(airspeed, z=z, original_data=True, resolution_density=10000)
    # print_SE_size_1D(airspeed, z=z, original_data=True,  # override=True,
    #                  # resolution_density=10000,
    #                  save_as=f"plot_103/bad-metric/plot103-2d-airspeed-{airspeed}-z={z}")

if False and __name__ == '__main__':
    airspeed = 6  # Only ints
    sensor = 2  # Numbering starts from zero

    airspeed_i = data_experiments.airspeed_index(airspeed)
    PercentageOfPointsInSE.signal_energies = \
        data_experiments.seTW[:, sensor, :, airspeed_i].T.flatten()
    PercentageOfPointsInSE.mode = MetricMode.Accuracy

    print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
                  metric_init=PercentageOfPointsInSE)
    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               metric_init=PercentageOfPointsInSE,
    #               save_as=f'plot_103/pauls-metric/plot103-3d-airspeed-{airspeed}-default-view')
    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               metric_init=PercentageOfPointsInSE,
    #               elev_azim=(90, 270),
    #               save_as=f'plot_103/pauls-metric/plot103-3d-airspeed-{airspeed}-on-top-view')

if False and __name__ == '__main__':
    airspeed = None  # DON'T CHANGE
    sensor = 2  # Numbering starts from zero

    signal_energies = data_experiments.seTW[:, sensor, :, :].T.flatten()
    # Correct this, you might be throwing data away
    signal_energies = signal_energies[signal_energies != 0]
    PercentageOfPointsInSE.signal_energies = signal_energies

    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               metric_init=PercentageOfPointsInSE)
    print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
                  metric_init=PercentageOfPointsInSE,
                  save_as=f'plot_103/pauls-metric/plot103-3d-airspeed-{airspeed}-default-view')
    print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
                  metric_init=PercentageOfPointsInSE,
                  elev_azim=(90, 270),
                  save_as=f'plot_103/pauls-metric/plot103-3d-airspeed-{airspeed}-on-top-view')

if True and __name__ == '__main__':
    airspeed = 6
    # metric_init can be: SESize, Coverage, Accuracy
    # other metrics include: Error, Quality (They are faulty!!)
    # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
    #               # override=True, resolution_density=3000,
    #               metric_init=Error)

    Quality.weight = 100
    metrics: List[Tuple[Type[SEMetric], str]] = \
        [(Coverage, 'coverage'), (Accuracy, 'accuracy'), (Error, 'error'),
         (Quality, 'comb-accu-error')]

    for metric, name in metrics:
        print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
                      metric_init=metric,
                      # override=True, resolution_density=3000,
                      # save_as=f'plot_103/{name}/plot103-{name}-3d-airspeed-{airspeed}-default-view'
                      )
        # print_SE_size(airspeed, zs_n=20, taus_n=20, original_data=True, is3d=True,
        #               elev_azim=(90, 270),
        #               metric_init=metric,
        #               save_as=f'plot_103/{name}/plot103-{name}-3d-airspeed-{airspeed}-on-top-view')

    # tau = 1 - 1/2**12  # Very close to 100%
    # z = 2
    # print_SE_size_1D(airspeed, tau=tau, original_data=True,
    #                  metric_init=Coverage)
    # print_SE_size_1D(airspeed, z=z, original_data=True,
    #                  metric_init=Coverage)
