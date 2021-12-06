from __future__ import print_function

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

# To check type check code before running. Use `mypy` to check code
if 'typing' in sys.modules:
    from typing import List, Tuple, Any, Optional

# Elkin utilities
import probability_101 as prob
import data_104 as data_experiments

# Loading data here as D and S are defined as global variables by Sam and Paul's Safety
# Envelopes
if True:  # loading using Elkin's utilities
    airspeed = 15
    sensor = 2  # Numbering starts from zero

    airspeed_i = data_experiments.airspeed_index(airspeed)

    meanSE = data_experiments.means_all[sensor, :, airspeed_i]
    stdSE = np.sqrt(data_experiments.vars_all[sensor, :, airspeed_i])
    AoAs_n = len(data_experiments.stall_prob[sensor])
    AoAs = list(range(AoAs_n))
    x_array = data_experiments.seTW[:, sensor, :, airspeed_i].T

else:  # Loading from a CSV (Paul's preprocessing)
    stdSE = np.genfromtxt('data/pauls_se/stdSE.csv', delimiter=',')
    meanSE = np.genfromtxt('data/pauls_se/meanSE.csv', delimiter=',')
    AoAs = np.genfromtxt('data/pauls_se/AoAs.csv', delimiter=',')
    # Taking the signal energy values as a matrix where each column is a different AoA and
    # each row is a different time
    x_array = np.genfromtxt('data/pauls_se/signalEnergy_AllAoA.csv', delimiter=',').T

D = []  # D(T)  instead of <mean, variance>, storing <mean, stdev> directly for now
# Creating D(T)
for i in range(0, len(meanSE)):
    D.append((meanSE[i], stdSE[i], AoAs[i]))

S = []  # S(T)
# Creating S(T)
for d in D:
    if d[2] >= 14:
        S.append(d)


# No stall sentinel
def Safe_sentinel(x, D, S, sigma):  # No stall sentinel - assuming normal distribution for now
    # type: (float, List[Tuple[float, float, int]], List[Tuple[float, float, int]], float) -> bool
    forall_flag = True
    exists_flag = False
    for d in S:
        if not (abs(x - d[0]) > sigma*d[1]):
            forall_flag = False
    D_exclusion_S = [y for y in D if y not in S]  # D(T)\S(T)
    for d in D_exclusion_S:
        # print "x=",x,"d[0]=",d[0],"d[1]=",d[1]
        if abs(x - d[0]) <= sigma*d[1]:
            exists_flag = True
    if forall_flag and exists_flag:
        return True
    return False  # no stall

# Stall sentinel


def SafeP_sentinel(x, D, S, sigma):  # Stall sentinel - assuming normal distribution for now
    # type: (float, List[Tuple[float, float, int]], List[Tuple[float, float, int]], float) -> bool
    forall_flag = True
    exists_flag = False
    D_exclusion_S = [y for y in D if y not in S]  # D(T)\S(T)
    for d in D_exclusion_S:
        if not (abs(x - d[0]) > sigma*d[1]):
            forall_flag = False
    for d in S:
        # print "x=",x,"d[0]=",d[0],"d[1]=",d[1]
        if abs(x - d[0]) <= sigma*d[1]:
            exists_flag = True
    if forall_flag and exists_flag:
        return True
    return False  # stall safe P


# Determining how many signal energies fall within the signal envelope for Elkin's Safety
# Envelopes
def main_ESE(use_sigma):
    # type: (float) -> Tuple[List[float], List[float], List[float], List[int]]
    z = use_sigma
    tau = norm.cdf(z) - norm.cdf(-z)

    n_stall = 14
    # Safety Envelopes Elkin
    se = prob.wing_SE_classification(meanSE, stdSE, n_stall, tau)

    tainted_aoa = []  # stores all the AoA that have values outside the envelope
    outside_signals = []  # for drawing the envelope boundaries
    inside_signals = []  # for drawing signals that fail
    all_signals = []  # for drawing the envelope boundaries

    for i in range(0, 18):
        x_row = x_array[i]
        for x in x_row:
            if x >= 2000:  # limiting to 2000 datapoints
                break
            all_signals.append(x)

            if se.inside_safety_envelope(x, z):
                inside_signals.append(x)
            else:
                outside_signals.append(x)
                tainted_aoa.append(i)

    return (all_signals, outside_signals, inside_signals, tainted_aoa)


# Determining which signal energies fall inside the envelope (Sam and Paul's envelope)
def main_SPSE(use_sigma):
    # type: (float) -> Tuple[List[float], List[float], List[float], List[int]]
    # x_axis = []
    # y_axis = []
    # results = []
    tainted_aoa = []  # stores all the AoA that have values outside the envelope
    outside_signals = []  # for drawing the envelope boundaries
    inside_signals = []  # for drawing signals that fail
    all_signals = []  # for drawing the envelope boundaries
    ################
    for i in range(0, 18):
        x_row = x_array[i]
        for x in x_row:
            if x >= 2000:  # limiting to 2000 datapoints
                break

            all_signals.append(x)
            safe_result = Safe_sentinel(x, D, S, use_sigma)
            safeP_result = SafeP_sentinel(x, D, S, use_sigma)
            # inside_envelope = True
            if not safe_result and not safeP_result:
                # inside_envelope = False
                outside_signals.append(x)
                tainted_aoa.append(i)
            else:
                inside_signals.append(x)
            # x_axis.append(i)  # AoA
            # y_axis.append(x)  # signal energy
            # results.append(inside_envelope)

    return (all_signals, outside_signals, inside_signals, tainted_aoa)


def metric1(use2_sigma):
    # type: (float) -> float
    result = main_SPSE(use2_sigma)
    # Calculating envelope metric:-
    # Formula - % of signal energies that return true for both Safe and Safe'
    no_coincide = 0
    for x in result[0]:
        coincides_flag = False
        if x in result[1]:
            coincides_flag = True
            # print("sss")
        if not coincides_flag:
            # print("aaa")
            no_coincide = no_coincide+1
    metric = (float(no_coincide) / float(len(result[0])))*100
    print("metric=", metric)
    return metric


def metric2(use3_sigma):
    # type: (float) -> List[int]
    result = main_SPSE(use3_sigma)[3]
    unique = list(dict.fromkeys(result))
    return unique


# Metric number of signal energies inside envelope
def metric3(use3_sigma, fmain=main_SPSE):
    # type: (float, Any) -> int
    result = fmain(use3_sigma)
    total_num = len(result[0])
    outside_num = len(result[1])
    inside_num = total_num-outside_num
    return inside_num


# percentage of of signal energies inside envelope
def metric4(use3_sigma):
    # type: (float) -> float
    result = main_SPSE(use3_sigma)
    total_num = len(result[0])
    outside_num = len(result[1])
    inside_num = total_num-outside_num
    # too small to see in graph, so graph comes linear
    percent = (inside_num*100)/total_num
    return percent


def plot_signal_energies_falling_inside_envelope(main, save_as=None):
    # type: (Any, Optional[str]) -> None
    sigma_array = np.linspace(0, 4, 81)
    metric_array = []

    for x in sigma_array:
        result = metric3(x, main)  # Metric: number of signal energies inside envelope
        # result = metric3(x, main_ESE)  # Metric: number of signal energies inside envelope
        # result = metric4(x) # Metric: percentage of signal energies inside envelope
        metric_array.append(result)
        print("Z: {x:.4f} Metric: {result:.2f}".format(x=x, result=result))

    plot_title = "Metric: Number of signal energies inside envelope"
    # plot_title ="Metric: Percentage of signal energies inside envelope"

    plt.figure()
    plt.title(plot_title)
    # mpl.rc("text", usetex="True")
    ax = plt.subplot(111)
    ax.grid(linestyle='-', linewidth='0.1', color='gray')
    plt.plot(sigma_array, metric_array, 'blue')
    # plt.scatter(sigma_array, metric_array, color='blue')
    # plt.bar(sigma_array, metric_array, align='center', alpha=0.5)
    plt.xlabel("$M_{\\sigma}$")  # , fontsize=22)
    plt.ylabel("$\\Psi$")  # , fontsize=22)
    # plt.xticks(np.arange(0, 4, 0.01), fontsize = 20)
    # plt.xticks(fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.yticks(fontsize=20)
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)


def plot_signal_energies_as_a_function_of_z(main, save_as=None):
    # type: (Any, Optional[str]) -> None
    sigma_array = np.linspace(0, 4, 81)
    signals = []

    for x in sigma_array:
        print("Z: {x:.4f}".format(x=x))
        # Metric: number of signal energies inside envelope
        all_signals, outside_signals, inside_signals, tainted_aoa = main(x)
        signals.append(inside_signals)

    mpl.rc("text", usetex="True")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.set_title("")
    ax.grid(linestyle='-', linewidth='0.1', color='gray')
    for i, x in enumerate(sigma_array):
        n_signals = len(signals[i])
        plt.scatter([x]*n_signals, signals[i], marker='_', color='blue')
    plt.xlabel("$M_{\\sigma}$")  # , fontsize=22)
    plt.ylabel("$\\Psi$")  # , fontsize=22)
    # plt.xticks(np.arange(0, 4, 0.01), fontsize = 20)
    # plt.xticks(fontsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.yticks(fontsize=20)
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)


if __name__ == '__main__':
    # plot_signal_energies_falling_inside_envelope(main_ESE)
    plot_signal_energies_falling_inside_envelope(
        main_ESE, save_as='plots/pauls_metric/paul_metric_ESE.png')
    plot_signal_energies_falling_inside_envelope(
        main_SPSE, save_as='plots/pauls_metric/paul_metric_SPSE.png')

    # plot_signal_energies_as_a_function_of_z(main_ESE)
    plot_signal_energies_as_a_function_of_z(
        main_ESE, save_as='plots/pauls_metric/signals_function_of_z_ESE.png')
    plot_signal_energies_as_a_function_of_z(
        main_SPSE, save_as='plots/pauls_metric/signals_function_of_z_SPSE.png')
