from __future__ import annotations

from typing import Tuple, Optional, Union

import numpy as np
import scipy.io as sio

# Preprocessing distributions
mat_contents = sio.loadmat('data/windTunnel_signalEnergy_data_win1s.mat')

seTW = mat_contents['seTW_filt']

AoAs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# computed total and stall dists, per airspeed
# total_all = [18, 18, 18, 18, 18, 18, 18, 18, 17, 16, 14, 13, 12, 9, 7]
# n_stall_all = [7, 7, 5, 5, 5, 5, 4, 4, 4, 4, 2, 2, 1, 0, 0]
stall_prob = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 21
    [0, 0, 0, 0, 0, 0, 0]  # 22
]

assert len(airspeeds) == len(stall_prob)

means_all = np.mean(seTW, axis=0)
vars_all = np.var(seTW, axis=0, ddof=1)


def dists_given_airspeed(
        airspeed: Optional[int],
        sensors: Union[int, Tuple[int, ...]] = 0,
        verbose: bool = True
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(sensors, int):
        assert 0 <= sensors < 8, \
            f"Sensor number `{sensors}` is incorrect. Only possible sensor values: 0-7"
    else:
        assert all(0 <= sensor < 8 for sensor in sensors), \
            f"One sensor number from `{sensors}` is incorrect. Only possible sensor values: 0-7"
        assert len(sensors) > 0, "There must be at least one sensor"

    if isinstance(sensors, tuple) and len(sensors) == 1:
        sensors = sensors[0]

    if airspeed in airspeeds:
        assert 6 <= airspeed <= 22, f"The airspeed {airspeed} is outside the range [6, 17]"

        airspeed_i = airspeeds.index(int(airspeed))
        stall = np.array(stall_prob[airspeed_i])
        total = stall.shape[0]
        means = means_all[sensors, :total, airspeed_i].T
        if isinstance(sensors, int):
            vars_ = vars_all[sensors, :total, airspeed_i]
        else:
            covs_ = []
            for data_AoA in seTW[:, sensors, :total, airspeed_i].transpose((2, 0, 1)):
                cov = np.cov(data_AoA)
                covs_.append(cov.reshape((1,) + cov.shape))
            vars_ = np.concatenate(covs_)
    elif airspeed is None:
        total = 0
        means_l = []
        stds_l = []
        stall_l = []
        # all airspeeds at the same time
        for airspeed in airspeeds:
            # using recursion to solve the problem instead of copying code
            total_i, means_i, stds_i, stall_i = \
                dists_given_airspeed(airspeed, sensors=sensors, verbose=False)
            total += total_i
            means_l.append(means_i)
            stds_l.append(stds_i)
            stall_l.append(stall_i)
        means = np.concatenate(means_l)
        vars_ = np.concatenate(stds_l)
        stall = np.concatenate(stall_l)
    else:
        raise Exception(f"Airspeed {airspeed} is not one of {airspeeds}")

    if verbose:
        print("Total number of angles of attack:", total)
        print(f"Means {means.shape}")
        print(means)
        print(f"Variances {vars_.shape}")
        print(vars_)
        print("Stall probabilities")
        print(stall)

    return total, means, vars_, stall


def get_shift_on_all_airspeeds(airspeed: int) -> int:
    total_all = [len(sp) for sp in stall_prob]
    airspeed_i = airspeeds.index(int(airspeed))
    return sum(total_all[:airspeed_i])


def dist_given_airspeed_AoA(
    airspeed: int,
    AoA: float
) -> Tuple[float, float]:
    total_AoAs, means, vars_, stall_prob = dists_given_airspeed(airspeed, verbose=False)

    if not (0 <= AoA <= total_AoAs):
        raise Exception(f"The angle of attack {AoA} is outside the range [1, 16]")

    AoA_integer = int(AoA)

    if (isinstance(AoA, int) or AoA.is_integer()):
        mean = means[AoA_integer]
        var = vars_[AoA_integer]
    else:
        w = AoA - AoA_integer
        mean = (1-w) * means[AoA_integer] + w * means[AoA_integer + 1]
        var = (1-w) * vars_[AoA_integer] + w * vars_[AoA_integer + 1]

    return mean, var


# The type for the result is incorrect it should be something like
# Union[nd.array, Tuple[nd.array, ...]]
def synthetize_signal_fixed_params(airspeed: int, AoA: float, size: int = 1) -> np.ndarray:
    mean, var = dist_given_airspeed_AoA(airspeed, AoA)
    gen = np.random.normal(loc=mean, scale=np.sqrt(var), size=1)
    return gen[0] if size == 1 else gen  # type: ignore


def synthetize_signal(airspeeds: np.ndarray, AoAs: np.ndarray) -> np.ndarray:
    return np.vectorize(synthetize_signal_fixed_params)(airspeeds, AoAs)  # type: ignore


def airspeed_index(airspeed: int) -> Optional[int]:
    return airspeeds.index(airspeed) if airspeed in airspeeds else None


if __name__ == '__main__':
    # dists_given_airspeed(8.7, verbose=True)
    # print(dist_given_airspeed_AoA(8.7, 10.4))
    # print(synthetize_signal_fixed_params(8.7, 10.4))
    # print(synthetize_signal(8 * np.ones((100,), dtype=int), 10.4 * np.ones((100,))))
    # dists_given_airspeed(10, sensors=0, verbose=True)
    dists_given_airspeed(10, sensors=(0,), verbose=True)
    dists_given_airspeed(10, sensors=(0, 1), verbose=True)
    # dists_given_airspeed(None, sensors=0, verbose=True)
    # dists_given_airspeed(None, sensors=(0, 1), verbose=True)
