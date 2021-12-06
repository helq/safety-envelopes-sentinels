from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import scipy.io as sio


wind_data = 'data/windTunnel_signalEnergy_data_win1s.mat'
data_n = 91
data_n_mean = data_n
seTW_name = 'seTW_filt'


def legacy_data(revert_to_bad: bool = False) -> None:
    """Calling this function makes the code behave as it did before, as it did in the
    paper. It is faulty because it is not using the filtered data, and because I made a
    grave mistake (not diving :S).
    """
    if not revert_to_bad:
        return
    global wind_data, data_n_mean, seTW_name
    wind_data = 'data/windTunnel_data_sensor3_AS15.mat'
    data_n_mean = 1
    seTW_name = 'seTW'


def dists_given_airspeed(airspeed: Optional[int], verbose: bool = True):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, int, int, List[int]]
    mat_contents = sio.loadmat(wind_data)

    seTW = mat_contents[seTW_name]

    # computed total and stall dists, per airspeed
    total_all = [18, 18, 18, 18, 18, 18, 18, 18, 17, 16, 14, 13, 12, 9, 7]
    n_stall_all = [7, 7, 5, 5, 5, 5, 4, 4, 4, 4, 2, 2, 1, 0, 0]

    if airspeed is not None:
        total = total_all[airspeed]
        means = np.sum(seTW[:, 3, :total, airspeed], axis=0) / data_n_mean
        stds = np.sqrt(
            np.sum((seTW[:, 3, :total, airspeed] - means.reshape((1, total)))**2, axis=0)
        ) / (data_n - 1)
        # total = 18  # means.size
        # n_stall = 5
        n_stall = n_stall_all[airspeed]
        stall_per_dist = [0]*(total - n_stall) + [1]*n_stall

    else:  # all airspeeds at the same time
        means_ = []
        stds_ = []
        n_stall = 0
        stall_per_dist = []
        for v_i, (t, n_s) in enumerate(zip(total_all, n_stall_all)):
            means = np.sum(seTW[:, 3, :t, v_i], axis=0) / data_n_mean
            stds = (np.sqrt(np.sum((seTW[:, 3, :t, v_i] - means.reshape((1, t)))**2, axis=0))
                    / (data_n - 1))
            means_.append(means)
            stds_.append(stds)
            n_stall += n_s
            # print(n_s)

            stall_per_dist += [0]*(t - n_s) + [1]*n_s

        means = np.concatenate(means_)
        stds = np.concatenate(stds_)

        total = means.size

    if verbose:
        print("Means")
        print(means)
        print("Standard deviations")
        print(stds)
        print("Total number of distributions:", total)
        print("Number of distributions in stall:", n_stall)

    return means, stds, total, n_stall, stall_per_dist


if __name__ == '__main__':
    airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # m/s

    for i, airspeed in enumerate(airspeeds):
        print(f"# airspeed = {airspeed}")
        means, stds, total, n_stall, stall_per_dist = dists_given_airspeed(i, verbose=False)

        print(f"means.append({means.tolist()})")
        print(f"stds.append({stds.tolist()})")
        print(f"stall_prob.append({stall_per_dist})")
