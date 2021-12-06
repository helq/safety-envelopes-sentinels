from __future__ import annotations

from typing import Optional, Tuple

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
    # type: (...) -> Tuple[np.ndarray, np.ndarray, int, int]
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

    else:  # all airspeeds at the same time
        means_red = []
        stds_red = []
        means_blue = []
        stds_blue = []
        for v_i, (t, n_s) in enumerate(zip(total_all, n_stall_all)):
            means = np.sum(seTW[:, 3, :t, v_i], axis=0) / data_n_mean
            stds = (np.sqrt(np.sum((seTW[:, 3, :t, v_i] - means.reshape((1, t)))**2, axis=0))
                    / (data_n - 1))
            if n_s == 0:
                means_red.append(means)
                stds_red.append(stds)
            else:
                means_red.append(means[:-n_s])
                means_blue.append(means[-n_s:])
                stds_red.append(stds[:-n_s])
                stds_blue.append(stds[-n_s:])
                # print(means_blue[-1].size)

        means_red_ = np.concatenate(means_red)
        stds_red_ = np.concatenate(stds_red)
        means_blue_ = np.concatenate(means_blue)
        stds_blue_ = np.concatenate(stds_blue)

        n_stall = stds_blue_.size
        means = np.concatenate([means_red_, means_blue_])
        stds = np.concatenate([stds_red_, stds_blue_])
        total = means.size

    if verbose:
        print("Means")
        print(means)
        print("Standard deviations")
        print(stds)
        print("Total number of distributions:", total)
        print("Number of distributions in stall:", n_stall)

    return means, stds, total, n_stall
