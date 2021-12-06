from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import scipy.io as sio
# import sys

__all__ = ['dists_given_airspeed', 'dist_given_airspeed_AoA',
           'synthetize_signal_fixed_params', 'synthetize_signal']

# Preprocessing distributions
mat_contents = sio.loadmat('data/windTunnel_signalEnergy_data_win1s.mat')

seTW = mat_contents['seTW_filt']

AoAs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# computed total and stall dists, per airspeed
# total_all = [18, 18, 18, 18, 18, 18, 18, 18, 17, 16, 14, 13, 12, 9, 7]
# n_stall_all = [7, 7, 5, 5, 5, 5, 4, 4, 4, 4, 2, 2, 1, 0, 0]
# The number of angles of attack and airspeed got truncated to make it possible, to
# have fixed number of angles of attack, and be able to interpolate between
# distributions.
# total_all =   [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]  # noqa: E222
# n_stall_all = [ 5,  5,  3,  3,  3,  3,  3,  4,  4,  4]  # noqa: E201
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

means_array = np.mean(seTW[:, 0, :, :], axis=0).T
stds_array = np.std(seTW[:, 0, :, :], axis=0, ddof=1).T

# code for legacy behaviour
total_AoAs: int
legacy_code = False


def legacy_data(revert_to_bad: bool = False) -> None:
    """Calling this function makes the code behave as it did before, as it did in the
    paper. It is faulty because it is not using the filtered data, and because I made a
    grave mistake (not diving :S).
    """
    global total_AoAs, mat_contents, seTW, means_array, stds_array, legacy_code, stall_prob
    assert not legacy_code, "The data has already been loadde with the legacy code"
    if not revert_to_bad:
        return
    total_AoAs = 16
    mat_contents = sio.loadmat('data/windTunnel_data_sensor3_AS15.mat')
    seTW = mat_contents['seTW']
    stall_prob = [p[:total_AoAs] for p in stall_prob[:10]]
    means_array = [np.sum(seTW[:, 3, :total_AoAs, i], axis=0) for i in range(10)]  # type: ignore
    stds_array = [  # type: ignore
        np.sqrt(
            np.sum(
                (seTW[:, 3, :total_AoAs, i] - means_array[i].reshape((1, total_AoAs)))**2,
                axis=0)
        ) / 90
        for i in range(10)
    ]
    legacy_code = True


def dists_given_airspeed(airspeed: Optional[float], verbose: bool = True):
    # type: (...) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]
    if airspeed is not None:
        assert 6 <= airspeed <= 22, f"The airspeed {airspeed} is outside the range [6, 17]"

    if airspeed is None:
        total_ = 0
        means_l = []
        stds_l = []
        stall_l = []
        # all airspeeds at the same time
        for airspeed in airspeeds:
            # using recursion to solve the problem instead of copying code
            total_i, means_i, stds_i, stall_i = \
                dists_given_airspeed(airspeed, verbose=False)
            total_ += total_i
            means_l.append(means_i)
            stds_l.append(stds_i)
            stall_l.append(stall_i)
        means = np.concatenate(means_l)
        stds = np.concatenate(stds_l)
        stall = np.concatenate(stall_l)
    elif (isinstance(airspeed, int) or airspeed.is_integer()) and int(airspeed) in airspeeds:
        airspeed_i = airspeeds.index(int(airspeed))
        stall = np.array(stall_prob[airspeed_i])
        total_ = stall.shape[0]
        means = means_array[airspeed_i][:total_]
        stds = stds_array[airspeed_i][:total_]
        # print("AIRSPEED:", airspeed_i)
    # elif isinstance(airspeed, float):
    else:
        assert legacy_code, "This code hasn't been modified to work in the fixed version"
        total_ = total_AoAs
        # Finding upper airspeed index for current airspeed
        airspeed_up: int = next(i for (i, airs) in enumerate(airspeeds) if airs > airspeed)
        airspeed_low = airspeed_up - 1
        low, up = airspeeds[airspeed_low], airspeeds[airspeed_up]
        w: float = (up - airspeed) / (up - low)
        # print("LOW/UP and WEIGTH:", airspeed_low, airspeed_up, w)

        # print([m.shape for m in means_array])
        means = w * means_array[airspeed_low] + (1-w) * means_array[airspeed_up]
        stds = w * stds_array[airspeed_low] + (1-w) * stds_array[airspeed_up]
        stall = w * np.array(stall_prob[airspeed_low]) \
            + (1-w) * np.array(stall_prob[airspeed_up])

    if verbose:
        print("Total number of angles of attack:", means.shape[0])
        print("Means")
        print(means)
        print("Standard deviations")
        print(stds)
        print("Stall probabilities")
        print(stall)

    return total_, means, stds, stall


def get_shift_on_all_airspeeds(airspeed: int) -> int:
    total_all = [len(sp) for sp in stall_prob]
    airspeed_i = airspeeds.index(int(airspeed))
    return sum(total_all[:airspeed_i])


def dist_given_airspeed_AoA(
    airspeed: float,
    AoA: float
) -> Tuple[float, float]:
    total_AoAs, means, stds, stall_prob = dists_given_airspeed(airspeed, verbose=False)

    if not (0 <= AoA <= total_AoAs):
        raise Exception(f"The angle of attack {AoA} is outside the range [1, 16]")

    AoA_integer = int(AoA)
    if legacy_code:
        AoA_integer -= 1
        AoA -= 1

    if (isinstance(AoA, int) or AoA.is_integer()):
        mean = means[AoA_integer]
        std = stds[AoA_integer]
    else:
        w = AoA - AoA_integer
        mean = (1-w) * means[AoA_integer] + w * means[AoA_integer + 1]
        std = (1-w) * stds[AoA_integer] + w * stds[AoA_integer + 1]

    return mean, std


# The type for the result is incorrect it should be something like
# Union[nd.array, Tuple[nd.array, ...]]
def synthetize_signal_fixed_params(airspeed: float, AoA: float, size: int = 1) -> np.ndarray:
    mean, std = dist_given_airspeed_AoA(airspeed, AoA)
    gen = np.random.normal(loc=mean, scale=std, size=1)
    return gen[0] if size == 1 else gen  # type: ignore


def synthetize_signal(airspeeds: np.ndarray, AoAs: np.ndarray) -> np.ndarray:
    return np.vectorize(synthetize_signal_fixed_params)(airspeeds, AoAs)  # type: ignore


if __name__ == '__main__':
    # dists_given_airspeed(8.7, verbose=True)
    # print(dist_given_airspeed_AoA(8.7, 10.4))
    # print(synthetize_signal_fixed_params(8.7, 10.4))
    print(synthetize_signal(8.7 * np.ones((100,)), 10.4 * np.ones((100,))))
