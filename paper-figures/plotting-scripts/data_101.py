from __future__ import annotations

from typing import Tuple, List, Optional

from pathlib import Path
import numpy as np
import csv

__all__ = ['dists_given_airspeed']

path_data = Path("data/Ahmad - GPRMs")

AoAs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
airspeeds = [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

stall_prob: List[List[float]] = [
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
# total = 16

# read csv with artificial AoA
with open(path_data / "Artificial_AoA_Vector.csv", "r", newline='') as csvfile:
    artificialAoAs = [float(line[0]) for line in csv.reader(csvfile)]


def dists_given_airspeed(airspeed: int, skip: int = 0, verbose: bool = True):
    # type: (...) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    assert airspeed in airspeeds, f"The airspeed {airspeed} is not in the 'database'"

    # read momenta from csv for a specific airspeed
    with open(path_data / f"AS{airspeed}.csv", "r", newline='') as csvfile:
        momenta = np.array([(float(line[0]), float(line[1]))
                            for line in csv.reader(csvfile)])
    means = momenta[:, 0]
    vars_ = momenta[:, 1]
    AoAs = np.array(artificialAoAs)[:means.shape[0]]

    if skip > 1:
        AoAs = AoAs[::skip]
        means = means[::skip]
        vars_ = vars_[::skip]

    # generate "stall" information (extend it to cover all artificial AoAs)
    stall_for_airspeed = stall_prob[airspeeds.index(airspeed)]
    stall = np.array([interpolate_stall(stall_for_airspeed, AoA)
                      for AoA in AoAs])

    return AoAs, means, vars_, stall


def interpolate_stall(stall_prob: List[float], AoA: float) -> float:
    if 15 - AoA < 10e-3:
        return stall_prob[15]

    low_i = int(AoA)
    up_i = low_i + 1
    prob = AoA - low_i
    return stall_prob[low_i] * (1 - prob) + stall_prob[up_i] * prob


def reload_global_data(path_data_: Optional[str] = None) -> None:
    global artificialAoAs
    global path_data
    if path_data_ is not None:
        path_data = Path(path_data_)

    with open(path_data / "Artificial_AoA_Vector.csv", "r", newline='') as csvfile:
        artificialAoAs = [float(line[0]) for line in csv.reader(csvfile)]
