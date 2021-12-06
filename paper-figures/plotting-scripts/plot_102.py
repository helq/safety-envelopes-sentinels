from __future__ import annotations

from typing import Optional, Dict, Any, Iterable, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation

import data_104 as data_experiments
# import data_005 as data
import data_101 as data_artificial
import probability_101 as prob
# from color import color_interpolate
from misc import region_from_positives, color_interpolate


dirname = "plots/plot_102"


# old plot from plot_003
def plot_top_bars(
    ax: Any,
    means: np.ndarray,
    stds: np.ndarray,
    z: float,
    stall_per_dist: Iterable[float],
    yticks_sep: Optional[int] = 2,
    anglesofattack: Optional[np.ndarray] = None,
    start_at_zero: bool = False
) -> None:
    if anglesofattack is None:
        anglesofattack = np.arange(0, stds.size) if start_at_zero else \
                         np.arange(1, stds.size+1)

    # States bars
    color_bars = [color_interpolate((0x46, 0x77, 0xc8), (0xc8, 0x5c, 0x46), d)
                  for d in stall_per_dist]
    ax.barh(anglesofattack, 2*z*stds, left=means-z*stds, height=.7,
            align='center', color=color_bars, zorder=2)

    # Points on top of the state bars
    ax.scatter(means, anglesofattack, color='#000000', zorder=4)
    if yticks_sep is None:
        ax.set_yticks([])
    elif yticks_sep > 0:
        ax.set_yticks(anglesofattack[yticks_sep-1::yticks_sep])
        ax.set_yticklabels(anglesofattack[yticks_sep-1::yticks_sep])


# This plots a countour instead of bars for the angles of attack
def plot_top_contour(
    ax: Any,
    anglesofattack: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    z: float,
    yticks_sep: Optional[int] = 2,
) -> None:
    # z-confident region for each distribution
    poly = Polygon([*zip(means-z*stds, anglesofattack),
                    *reversed(list(zip(means+z*stds, anglesofattack)))])
    ax.add_patch(poly)

    # mean of distribution
    ax.plot(means, anglesofattack, color='#000000', zorder=4)

    if yticks_sep is None:
        ax.set_yticks([])
    elif yticks_sep > 0:
        ax.set_yticks(anglesofattack[yticks_sep-1::yticks_sep])
        ax.set_yticklabels(anglesofattack[yticks_sep-1::yticks_sep])


def plot_predictability_region(
    ax: Any,
    xs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    height: float,
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

    fst_cmp = np.array([1, 0]).reshape((1, 2))
    snd_cmp = np.array([0, 1]).reshape((1, 2))
    region1_verts = np.array(list(region_from_positives(zip(xs, region1))))
    val_bumps = region1_verts * snd_cmp * 0.6 * height - snd_cmp * height
    pos_bumps = region1_verts * fst_cmp
    region1_verts = val_bumps + pos_bumps

    region1_poly = Polygon(list(region1_verts), facecolor='#777', edgecolor='#777')
    ax.add_patch(region1_poly)
    ax.set_ylim(bottom=-height)

    return region1  # type: ignore


def fig_plot(  # noqa: C901
    z: float,
    tau: float,
    airspeed: Optional[int],
    save_as: Optional[str] = None,
    latex: bool = False,
    subplot_adjust_params: Optional[Dict[str, float]] = None,
    undertitle: bool = True,
    bars: bool = False,
    skip: int = 0,
    original_data: bool = False,
    force_left_plot: Optional[float] = None,
    force_right_plot: Optional[float] = None
) -> None:
    if original_data:
        assert skip == 0, \
            "`skip` should only be defined when the `original_data` is not used"

    if save_as and latex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 10
        })
    else:
        matplotlib.use('Qt5Agg')

    if original_data:
        total, means, vars_, stall = data_experiments.dists_given_airspeed(airspeed)
    else:
        assert isinstance(airspeed, int)
        AoAs, means, vars_, stall = data_artificial.dists_given_airspeed(airspeed, skip=skip)
    stds = np.sqrt(vars_)

    left_plot = (means - z*stds).min() if force_left_plot is None else force_left_plot
    right_plot = (means + z*stds).max() if force_right_plot is None else force_right_plot

    shift_x = 1.1
    plot_density = 1000

    ##################################
    # Plotting prep
    # xs = np.arange(left_plot, right_plot, 1)
    xs = np.arange(left_plot, shift_x*right_plot, (right_plot-left_plot)/plot_density)

    ##################################
    # Plotting classification results
    se = prob.wing_SE_classification(means, stds, stall, tau)

    # PLOTTING
    ##################################
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9), sharex='col', gridspec_kw={'height_ratios': [12, 12, 1.6]})
    # ax.set_xlabel('Signal Energy')
    if airspeed is None:
        ax.set_title("All airspeeds")
    else:
        ax.set_title(f"\\(v = \\) {airspeed} m/s" if latex else f"v = {airspeed} m/s")

    ax.set_ylabel('Angle of Attack')
    ax2.set_ylabel('P[stall | X=x]')
    ax3.set_ylabel('SE')
    if undertitle:
        ax3.set_xlabel(r'Signal Energy $(V^2 \cdot s)$' if latex
                       else 'Signal Energy (V^2 s)')

    # Plot 1
    # This line seems to not be necessary in the latest version of `prob` code,
    # it can be replaced with the data obtained from data_005
    # means_, stds_, _, _, stall_per_dist = data_003.dists_given_airspeed(airspeed, False)
    if bars or original_data:
        if 'total' not in vars():
            assert 'AoAs' in vars()
            plot_top_bars(ax, means, stds, z, stall,
                          yticks_sep=0, anglesofattack=AoAs)
            region1 = plot_predictability_region(ax, xs, means, stds, float(AoAs.max()), z)
        else:
            plot_top_bars(ax, means, stds, z, stall, yticks_sep=0,
                          start_at_zero=True)
            region1 = plot_predictability_region(ax, xs, means, stds, total/6, z)
    else:
        plot_top_contour(ax, AoAs, means, stds, z, yticks_sep=0)
        bar_height = float(AoAs.max() - AoAs.min()) / 6
        region1 = plot_predictability_region(ax, xs, means, stds, bar_height, z)
    # ax.set_yticks([])
    if force_left_plot is not None:
        ax.set_xlim(left=left_plot)
    if force_right_plot is not None:
        ax.set_xlim(right=shift_x*right_plot)

    # Plot 2
    ax2.plot(xs, np.vectorize(se.pdf_red)(xs), color='black')
    region2 = se.plot_certainty_region(ax2, xs, 0.09, '0.2', shift_y=0.13)

    # pdf = []
    # for s in xs:
    #     pdf.append(se.pdf_red(s))
    #     post = se.posterior_for_states(s)
    #     print(s, sum(post), post)
    #     se.p_dist = prob.ArbitraryDiscreteDistribution({
    #         i: p for i, p in enumerate(post)
    #     })

    # ax2.plot(xs, pdf, color='black')
    # region2 = se.plot_certainty_region(ax2, xs, 0.09, '0.2', shift_y=0.13)

    # Plot 3
    assert isinstance(region2, np.ndarray)
    region = region1 * region2
    region_verts = region_from_positives(zip(xs, region))
    region_poly = Polygon(list(region_verts), facecolor='#60af3d', edgecolor='#fff')
    ax3.set_yticks([])
    ax3.add_patch(region_poly)
    ax3.set_ylim(bottom=0.03, top=1)

    # from matplotlib.ticker import MaxNLocator
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

    # plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.1, hspace=0, wspace=0)
    if subplot_adjust_params is not None:
        params = {'top': 1, 'bottom': 0.05, 'right': 1, 'left': 0.1, 'hspace': 0, 'wspace': 0}
        params.update(subplot_adjust_params)
        plt.subplots_adjust(**params)

    if save_as:
        plt.savefig(f'{dirname}/{save_as}.png')
        if latex:
            plt.savefig(f'{dirname}/{save_as}.pgf')
        else:
            plt.savefig(f'{dirname}/{save_as}.svg')
        print(f'Plots saved to: `{dirname}/{save_as}`')
    else:
        plt.show()


def plot_stall_prob(ax: Any, xs: np.ndarray, p_stall: np.ndarray, tau: float) -> None:
    ax.plot(xs, p_stall)

    bottom = -0.25
    height = 0.15
    blue_region = np.array(list(region_from_positives(zip(xs, (1 - p_stall) > tau))))
    blue_region *= np.array([1, height]).reshape((1, 2))
    blue_region += np.array([0, bottom]).reshape((1, 2))
    blue_poly = Polygon(list(blue_region), facecolor='#4677c8', edgecolor='#00000000')
    ax.add_patch(blue_poly)

    red_region = np.array(list(region_from_positives(zip(xs, p_stall > tau))))
    red_region *= np.array([1, height]).reshape((1, 2))
    red_region += np.array([0, bottom]).reshape((1, 2))
    red_poly = Polygon(list(red_region), facecolor='#c85c46', edgecolor='#fff')
    ax.add_patch(red_poly)

    ax.set_ylim(bottom=bottom, top=1.05)


def simulate_posterior_change(  # noqa: C901
    airspeed: Union[int, None, Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    signal_energy_inputs: np.ndarray,
    tau: float,
    actual_states: Optional[np.ndarray] = None,
    sensors: Union[Tuple[int, ...], int] = 0,
    original_data: bool = False,
    skip: int = 0,
    save_as: str = "",
    latex: bool = False,
    verbose: bool = False
) -> None:
    if save_as and latex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 10
        })
    else:
        matplotlib.use('Qt5Agg')

    # Passing custom data for debugging
    if isinstance(airspeed, tuple):
        total, means, vars_, stall = airspeed

    elif original_data:
        assert skip == 0, \
            "`skip` should only be defined when the `original_data` is not used"
        total, means, vars_, stall = \
            data_experiments.dists_given_airspeed(airspeed, sensors=sensors)
    elif isinstance(airspeed, int):
        assert sensors == 0, "Multiple sensors not implemeneted for artificial data"
        AoAs, means, vars_, stall = data_artificial.dists_given_airspeed(airspeed, skip=skip)
    else:
        raise Exception("The type of airspeed is neither: int, None nor the data")
    # stds = np.sqrt(vars_)

    se_control = prob.wing_SE_classification(means, vars_, stall, tau, variance=True)
    se = prob.wing_SE_classification(means, vars_, stall, tau, variance=True)

    p_stall = []
    p_stall_updating_prior = []
    p_states = []
    for s in signal_energy_inputs:
        # Computing posterior probabilities
        post = se.posterior_for_states(s)
        p_states.append(post)

        # Updating SE classification with new probability distribution
        se.p_dist = prob.ArbitraryDiscreteDistribution({
            i: p for i, p in enumerate(post)
        })
        p_stall.append(se_control.pdf_red(s))
        p_stall_updating_prior.append(se.pdf_red(s))

        if verbose:
            print(f"signal: {s}  most likely state: {post.argmax()}  "
                  f"stall probability: {se.pdf_red(s)}")
            print("probabilities: [", end='')
            for p in post:
                print(f"{p:.3f} ", end='')
            print("]")

    p_states_ = np.array(p_states)
    states_n = p_states_.shape[1]
    xs = np.arange(len(p_stall))

    # modifying p_states to make it look nicer in the plot
    p_states_nice = p_states_
    # p_states_nice = p_states_ / p_states_.max(axis=1).reshape((-1, 1))

    print("Final state")
    print(f"signal: {s}  most likely state: {post.argmax()}  "
          f"stall probability: {se.pdf_red(s)}")

    # Plotting
    fig, (ax1, ax2, ax3, ax4) = \
        plt.subplots(4, 1, figsize=(10, 9), sharex='col',
                     gridspec_kw={'height_ratios': [3, 2, 2, 8]})
    ax1.set_title('Updating prior in classification and its change')
    ax1.set_ylabel('Signal Energy')
    ax2.set_ylabel('P[stall|X=x]\nuniform')
    ax3.set_ylabel('P[stall|X=x]\nupdating')
    ax4.set_ylabel('Posterior probabilites\nper each state')
    ax4.set_xlabel('Time (s)')

    # Plot 1: Signal energy plotting
    ax1.plot(xs, signal_energy_inputs)

    # Plot 4: Probability of stall plotting
    plot_stall_prob(ax2, xs, np.array(p_stall), tau)

    # Plot 3: Probability of stall plotting
    plot_stall_prob(ax3, xs, np.array(p_stall_updating_prior), tau)

    # Plot 4: Current probability for each state
    color_states = [color_interpolate((0x46, 0x77, 0xc8), (0xc8, 0x5c, 0x46), d)
                    for d in stall]

    # Change of probability for each state
    for state_i in range(states_n):
        # ax4.plot(xs, p_states_[:, state_i])
        p_state_i_height = p_states_nice[:, state_i] * .7
        poly = Polygon([*zip(xs, state_i + p_state_i_height),
                        *reversed(list(zip(xs, state_i - p_state_i_height)))],
                       color=color_states[state_i])
        ax4.add_patch(poly)

    if actual_states is not None:
        ax4.plot(xs, actual_states, color='black')

    ax4.set_ylim(bottom=-.8, top=states_n)

    # Saving or showing
    if save_as:
        plt.savefig(f'{dirname}/{save_as}.png')
        if latex:
            plt.savefig(f'{dirname}/{save_as}.pgf')
        else:
            plt.savefig(f'{dirname}/{save_as}.svg')
        print(f'Plots saved to: `{dirname}/{save_as}`')
    else:
        plt.show()


def grid_prob_state_flight_change_animation(  # noqa: C901
    airspeed: Union[None, Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    signal_energy_inputs: np.ndarray,
    tau: float,
    # actual_states: Optional[np.ndarray] = None,
    sensors: Union[Tuple[int, ...], int] = 0,
    save_as: str = "",
    latex: bool = False,
    size: Optional[int] = None,
    verbose: bool = False
) -> None:
    if save_as and latex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
            'font.size': 10
        })
    else:
        matplotlib.use('Qt5Agg')

    # Passing custom data for debugging
    if isinstance(airspeed, tuple):
        total, means, vars_, stall = airspeed
    else:
        total, means, vars_, stall = \
            data_experiments.dists_given_airspeed(airspeed, sensors=sensors)
    # stds = np.sqrt(vars_)

    se = prob.wing_SE_classification(means, vars_, stall, tau, variance=True)

    p_stall_updating_prior = []
    p_states = []
    for s in signal_energy_inputs:
        # Computing posterior probabilities
        post = se.posterior_for_states(s)
        p_states.append(post)

        # Updating SE classification with new probability distribution
        se.p_dist = prob.ArbitraryDiscreteDistribution({
            i: p for i, p in enumerate(post)
        })
        p_stall_updating_prior.append(se.pdf_red(s))

        if verbose:
            print(f"signal: {s}  most likely state: {post.argmax()}  "
                  f"stall probability: {se.pdf_red(s)}")
            print("probabilities: [", end='')
            for p in post:
                print(f"{p:.3f} ", end='')
            print("]")

    p_states_ = np.array(p_states)
    states_n = p_states_.shape[1]

    # modifying p_states to make it look nicer in the plot
    p_states_nice = p_states_

    print("Final state")
    print(f"signal: {s}  most likely state: {post.argmax()}  "
          f"stall probability: {se.pdf_red(s)}")

    # Plotting
    figGrid = GridProbPLot(states_n)
    frames = 10
    ani = animation.FuncAnimation(
        figGrid.fig, figGrid.plot_grid, zip(signal_energy_inputs, p_states_nice),
        interval=frames, repeat=False, save_count=size)

    # Saving or showing
    if save_as:
        ani.save(f"{dirname}/{save_as}.m4v", animation.FFMpegWriter(fps=20))
        print(f'Plots saved to: `{dirname}/{save_as}`')
    else:
        plt.show()


class GridProbPLot:
    def __init__(self, states_n: int) -> None:
        # self.fig, self.ax = plt.subplots()
        self.states_n = states_n
        self.t = 0
        self.fig, (self.ax1, self.ax2) = \
            plt.subplots(2, 1,
                         gridspec_kw={'height_ratios': [1, 5]})
        self.signals = []  # type: List[np.ndarray]

    def plot_grid(self, input: Tuple[np.ndarray, np.ndarray]) -> None:
        self.t += 1
        if self.t % 10 == 0:
            print(f"Iteration {self.t}")

        signal, prob_states = input
        self.signals.append(signal)

        self.ax1.cla()
        self.ax2.cla()

        self.ax1.set_title(f'Final probabilities. t = {self.t}')
        self.ax1.set_xlabel('t')
        self.ax1.set_ylabel('Signal Energy')
        self.ax2.set_xlabel('Airspeeds')
        self.ax2.set_ylabel('AoAs')

        self.ax1.plot(self.signals)

        # Last probability for each state
        ccs = []
        i = 0
        for airs, stall_airpeeds in zip(data_experiments.airspeeds, data_experiments.stall_prob):
            for aoa, stall_p in zip(data_experiments.AoAs, stall_airpeeds):
                rad = np.sqrt(prob_states[i])
                color = color_interpolate((0x46, 0x77, 0xc8), (0xc8, 0x5c, 0x46), stall_p)
                ccs.append(plt.Circle((airs, aoa), rad, color=color))
                i += 1
        assert self.states_n == i

        self.ax2.set_aspect(1)
        self.ax2.set_ylim(-0.7, 17.7)
        self.ax2.set_xlim(5.3, 22.7)
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        for cc in ccs:
            self.ax2.add_artist(cc)


def get_signal_data(
    airspeed: int, AoAs_js: List[int], sensors: Union[int, Tuple[int, ...]] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    import scipy.io as sio
    mat_contents = sio.loadmat('data/windTunnel_signalEnergy_data_win1s.mat')
    # seTW = mat_contents['seTW']
    seTW = mat_contents['seTW_filt']
    airspeed_i = data_experiments.airspeeds.index(int(airspeed))

    AoAs: List[float] = []
    xs: Any = []
    for j in AoAs_js:
        new_xs = list(seTW[:, sensors, j, airspeed_i])
        xs += new_xs
        AoAs += [j]*len(new_xs)

    return xs, np.array(AoAs)


if False and __name__ == '__main__':
    tau = .99
    z = 3
    airspeed = 13

    fig_plot(z, tau, airspeed, undertitle=False, original_data=True, force_left_plot=0)
    # fig_plot(z, tau, airspeed, undertitle=False, bars=False, original_data=False,
    #          skip=100)
    # fig_plot(z, tau, airspeed, undertitle=False, bars=False, original_data=False,
    #          skip=100, force_left_plot=0)
    # fig_plot(z, tau, airspeed, undertitle=False, bars=True, original_data=True,
    #          save_as=f'SE/safety_envelopes_bare_airspeed={airspeed}m_'
    #                  f'original_z={z}_tau={tau}')

    fig_plot(z, tau, airspeed=None, undertitle=False, original_data=True,
             force_left_plot=0, force_right_plot=20)
    # fig_plot(z, tau, airspeed=None, undertitle=False, original_data=True,
    #          force_left_plot=0, force_right_plot=20,
    #          save_as=f'SE/safety_envelopes_bare_airspeed=ALL_'
    #                  f'original_z={z}_tau={tau}')


# Testing with made up data
if False and __name__ == '__main__':
    # means = np.array([[0, 0], [5, 1], [3, 3]])
    # vars_ = np.array([
    #     [[1, 0.1],
    #      [0.1, 1]],
    #     [[.5, 0.1],
    #      [0.1, .5]],
    #     [[1, -0.2],
    #      [-0.2, 1]],
    # ])
    # stall = np.array([0, 0, 1])
    means = np.array([[0, .1], [.1, 0]])
    # means = np.array([[0.05116362, 0.22155727], [0.05787366, 0.2520828]])
    vars_ = np.array([
        [[0.00026003, 0.00120182],
         [0.00120182, 0.00601781]],
        [[0.00036444, 0.00173092],
         [0.00173092, 0.00884931]]
    ])
    stall = np.array([0, 1])
    data = (2, means, vars_, stall)
    xs = []
    for i in range(20):
        xs.append(np.random.multivariate_normal(means[i % 2], vars_[i % 2], 10))
    tau = 0.99

    print(means.shape)
    print(vars_.shape)
    print(stall.shape)
    print(np.concatenate(xs).shape)
    simulate_posterior_change(data, np.concatenate(xs), tau,
                              original_data=True, verbose=True)

if True and __name__ == '__main__':
    # Things to try/show:
    # 1. sensors = 0
    # 2. sensors = (0, 1)
    # 3. sensors = (0, 6)
    # 4. all sensors
    sensors = 0  # type: Union[int, Tuple[int, ...]]
    # sensors = (0, 1, 6)
    # sensors = (0, 1)
    # sensors = tuple(range(8))
    tau = .99
    z = 3
    airspeed = 16

    AoAs_js = [15]
    # AoAs_js = [0, 1]
    # AoAs_js = [12, 13]
    # AoAs_js = list(range(18))
    # AoAs_js = list(range(10, 18))
    # AoAs_js = list(range(6, -1, -1))
    xs, AoAs = get_signal_data(airspeed, AoAs_js, sensors=sensors)

    # synthetizing data - Not realistic!
    # airspeeds_ = airspeed * np.ones((600,))
    # AoAs = np.linspace(AoAs_js[0], AoAs_js[-1], airspeeds_.size)
    # xs = data_007.synthetize_signal(airspeeds_, AoAs)

    if True:
        # Pretty powerful stuff!
        simulate_posterior_change(airspeed, np.array(xs), tau, actual_states=AoAs,
                                  sensors=sensors, original_data=True)
        # simulate_posterior_change(
        #     airspeed, xs, tau, actual_states=AoAs, original_data=True,
        #     latex=False, verbose=False,
        #     save_as=f"posterior/post-prob_airspeed={airspeed}-tau={tau}-z={z}-"
        #             f"AoAs=[{','.join([str(a) for a in AoAs_js])}]"
        #             f"-sensors={sensors}")

    # This takes a bit but it is wonderful
    if False:
        shift = data_experiments.get_shift_on_all_airspeeds(airspeed)
        simulate_posterior_change(None, xs, tau, actual_states=AoAs+shift,
                                  sensors=sensors, original_data=True)
        # simulate_posterior_change(
        #     None, xs, tau, actual_states=AoAs+shift, original_data=True,
        #     latex=False, verbose=False,
        #     save_as=f"posterior/post-prob_airspeed=ALL-tau={tau}-z={z}-"
        #             f"AoAs=[{','.join([str(a) for a in AoAs_js])} for airspeed={airspeed}]"
        #             f"-sensors={sensors}")

if False and __name__ == '__main__':
    # sensors = 0
    sensors = (0, 1, 6)
    # sensors = tuple(range(8))
    tau = .99
    z = 3
    airspeed = 15

    # import data.varying_state_data as data_varying
    # xs = data_varying.seTW_filt[:, sensors]

    import scipy.io as sio
    mat_contents = sio.loadmat('data/windTunnel_signalEnergy_data_win1s.mat')
    # seTW = mat_contents['seTW']
    xs = mat_contents['seTW_filt'][:, sensors]

    if True:
        simulate_posterior_change(airspeed, xs, tau, sensors=sensors, original_data=True)

    if True:
        simulate_posterior_change(None, xs, tau, sensors=sensors, original_data=True)
        # shift = data_experiments.get_shift_on_all_airspeeds(airspeed)
        # simulate_posterior_change(None, xs, tau, actual_states=AoAs+shift,
        #                           sensors=sensors, original_data=True)

if False and __name__ == '__main__':
    sensors = 0
    # sensors = (0, 1, 2)
    # sensors = tuple(range(8))
    tau = .99
    z = 3
    airspeed = 15

    if True:
        # AoAs_js = [0]
        # AoAs_js = [0, 1]
        # AoAs_js = [12, 13]
        AoAs_js = list(range(18))
        # AoAs_js = list(range(10, 18))
        # AoAs_js = list(range(6, -1, -1))
        xs, _ = get_signal_data(airspeed, AoAs_js, sensors=sensors)

        # xs = np.array(xs)[:30, :]
        # grid_prob_state_flight_change_animation(None, xs, tau, sensors=sensors)
        grid_prob_state_flight_change_animation(
            None, xs, tau, sensors=sensors, size=len(xs),
            save_as=f"grid/concatenating_AoA_0_to_19_vel_15m_sensors={sensors}"
        )

    # This requires to modify ['seTW_filt'] to ['seTW'] inside "data_104"
    if False:
        # import data.varying_state_data as data_varying
        # xs = data_varying.seTW[:, sensors]

        import scipy.io as sio
        mat_contents = sio.loadmat('data/windTunnel_signalEnergy_data_win1s.mat')
        # seTW = mat_contents['seTW']
        xs = mat_contents['seTW'][:, sensors]

        # grid_prob_state_flight_change_animation(None, xs, tau, sensors=sensors)
        grid_prob_state_flight_change_animation(
            None, xs, tau, sensors=sensors, size=len(xs),
            save_as=f"grid/varying_AoA_0_to_19_vel_15m_sensors={sensors}"
        )
