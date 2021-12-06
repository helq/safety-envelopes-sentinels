from __future__ import annotations

# from debug import interact

from typing import Optional, Union, Tuple, List, Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation

import plot_003
# import data_003
import data_007 as data
import probability_005 as prob


class SEPlot(object):
    def __init__(
        self,
        z: float,
        tau: float,
        undertitle: bool = True,
        right_limit: Optional[float] = None,
        mode: str = 'both',
        tick_time: float = 0.02,
        time_width: float = 10.0
    ) -> None:
        self.z = z
        self.tau = tau
        self.undertitle = undertitle
        self.right_limit = right_limit
        self.mode = mode

        if mode == 'both':
            self.fig, axes = \
                plt.subplots(3, 2, figsize=(12, 9), sharex='col',
                             gridspec_kw={'height_ratios': [12, 12, 1.6],
                                          'width_ratios': [1, 1]})
            self.left_axes = axes[:, 0]
            self.right_axes = axes[:, 1]
            self.left_twin_ax = self.left_axes[0].twinx()
        elif mode == 'right':
            self.fig, self.right_axes = \
                plt.subplots(3, 1, figsize=(6, 9), sharex='col',
                             gridspec_kw={'height_ratios': [12, 12, 1.6]})
        elif mode == 'left':
            self.fig, self.left_axes = \
                plt.subplots(3, 1, figsize=(6, 9), sharex='col',
                             gridspec_kw={'height_ratios': [1, 1, 1]})
            self.left_twin_ax = self.left_axes[0].twinx()
        else:
            raise Exception(f"There is no mode {mode}. Must be one of {{left, right, both}}")

        # Variables to use for right plot
        self.shift_x = 1.1
        self.left_plot = 0
        self.plot_density = 1000

        # Variables to use for left plot
        self.signal: List[float] = []
        self.airspeeds: List[float] = []
        self.angles_attack: List[float] = []
        self.probabilities: List[float] = []
        self.is_in_SE: List[float] = []
        self.ts: List[float] = []
        self.tick = 0
        self.tick_time = tick_time
        self.time_width = time_width

        self.init_plot()

    def init_plot(self) -> None:
        if self.mode in {'right', 'both'}:
            self.init_plot_right()
        if self.mode in {'left', 'both'}:
            self.init_plot_left()

    def init_plot_left(self, left: float = 0, right: Optional[float] = None) -> None:
        if right is None:
            right = self.time_width

        padding_proportion = 0.02
        padding = (right-left)*padding_proportion
        self.left_axes[0].set_xlim(left=left-padding, right=right+padding)

        self.left_axes[0].set_ylim(bottom=5.5, top=18.5)
        self.left_axes[2].set_ylim(bottom=-0.1, top=1.1)

        if hasattr(self, 'right_axes'):
            bot, top = self.right_axes[0].get_ylim()
            ticks = self.right_axes[0].get_yticks()
            self.left_twin_ax.set_ylim(bottom=bot, top=top)
            self.left_twin_ax.set_yticks(ticks)
        else:
            self.left_twin_ax.set_ylim(bottom=0, top=16)

        self.left_axes[0].set_ylabel('Airspeed', color='tab:red')
        self.left_twin_ax.set_ylabel('Angle of attack', color='tab:blue')
        self.left_axes[1].set_ylabel('Synthetized signal energy (V² s)')
        self.left_axes[2].set_ylabel('Probability of stall')
        self.left_axes[2].set_xlabel('Time')

    def init_plot_right(self) -> None:
        self.right_axes[0].set_yticks([])
        if self.mode != 'both':
            self.right_axes[0].set_ylabel('Angle of Attack')
        self.right_axes[1].set_ylabel('P[stall | X=x]')
        self.right_axes[2].set_ylabel('SE')
        if self.undertitle:
            self.right_axes[2].set_xlabel(r'Signal Energy $(V^2 \cdot s)$')

    def set_boundaries_and_xs(
        self,
        means: np.ndarray,
        stds: np.ndarray
    ) -> np.ndarray:

        maxi = np.where(means == means.max())[0].flatten()[0]
        right_plot = means[maxi] + self.z*stds[maxi]

        if (self.right_limit is not None) and (self.right_limit < right_plot):
            right_plot = self.right_limit

        self.right_axes[1].set_ylim(bottom=-0.1, top=1.1)
        self.right_axes[0].set_xlim(left=self.left_plot, right=right_plot)

        return np.arange(self.left_plot, right_plot*self.shift_x,  # type: ignore
                         (right_plot-self.left_plot)/self.plot_density)

    def fig_plot(
        self,
        input_: Union[float, Tuple[float, float, float]]
    ) -> None:
        airspeed = input_[0] if isinstance(input_, tuple) else input_
        self.tick += 1

        print(f"\nCurrent values. Tick: {self.tick} Airspeed: {airspeed}")
        total, means, stds, n_stall = dists_params = data.dists_given_airspeed(airspeed)
        # Plotting prep
        se = prob.wing_BIWT(means, stds, n_stall, self.tau, 0)

        if self.mode == 'right':
            assert isinstance(input_, float)
            self.fig_plot_right(airspeed, dists_params, se)
        if self.mode == 'left':
            assert isinstance(input_, tuple)
            self.fig_plot_left(input_, se)
        elif self.mode == 'both':
            assert isinstance(input_, tuple)
            # airspeed = input_[0]
            self.fig_plot_left(input_, se)
            self.fig_plot_right(input_, dists_params, se)

    def fig_plot_left(
        self,
        parms: Tuple[float, float, float],
        se: prob.BayesianInferenceWithThreshhold
    ) -> None:
        ax, ax2, ax3 = self.left_axes
        ax_twin = self.left_twin_ax

        airspeed, angle_attack, signal_v = parms

        t = self.tick * self.tick_time

        ax.cla()
        ax_twin.cla()
        ax2.cla()
        ax3.cla()
        if t > self.time_width:
            self.init_plot_left(left=t-self.time_width, right=t)
        else:
            self.init_plot_left()

        self.airspeeds.append(airspeed)
        self.angles_attack.append(angle_attack)
        self.signal.append(signal_v)
        self.probabilities.append(se.pdf_blue(signal_v))
        self.ts.append(t)
        self.is_in_SE.append(se.inside_safety_envelope(signal_v, self.z))

        ax.plot(self.ts, self.airspeeds, color='tab:red')
        ax_twin.plot(self.ts, self.angles_attack, color='tab:blue')
        ax2.plot(self.ts, self.signal)
        ax3.plot(self.ts, self.probabilities, zorder=4)
        plot_safety_region(ax3, self.ts, self.is_in_SE)

        xleft, xright = ax.get_xlim()
        x_pos = .75*(xright - xleft) + xleft
        ax.text(x_pos, 6, f"AoA = {angle_attack:.2f}")
        ax.text(x_pos, 6.8, f"v = {airspeed:.2f}")

    def fig_plot_right(
        self,
        input_: Union[float, Tuple[float, float, float]],
        dists_params: Tuple[int, np.ndarray, np.ndarray, np.ndarray],
        se: prob.BayesianInferenceWithThreshhold
    ) -> None:
        ax, ax2, ax3 = self.right_axes
        if isinstance(input_, float):
            airspeed = input_
            signal_v: Optional[float] = None
            angle_attack: Optional[float] = None
        else:
            airspeed, angle_attack, signal_v = input_

        total, means, stds, n_stall = dists_params

        ax.cla()
        ax2.cla()
        ax3.cla()

        self.init_plot_right()

        # Computing xs for this current plot
        xs = self.set_boundaries_and_xs(means, stds)

        # DYNAMIC PLOT HERE
        ##################################
        # ax.set_xlabel('Signal Energy')
        ax.set_title(f"v = {airspeed:.3f} m/s")

        # Plot 1
        plot_003.plot(ax, means, stds, total, self.z, n_stall)

        # Plot 2
        ax2.plot(xs, np.vectorize(se.pdf_blue)(xs), color='blue', zorder=2)
        region2 = se.plot_certainty_region(ax2, xs, 0.09, '0.2', shift_y=0.13)

        # Plot 3
        region1_ = xs.reshape((-1, 1)) * np.ones((xs.size, means.size))
        means__ = means.reshape((1, -1))
        stds__ = stds.reshape((1, -1))
        region1_ = np.bitwise_and(means__-self.z*stds__ < region1_,
                                  region1_ < means__+self.z*stds__)
        region1 = region1_[:, 0]
        for i in range(1, means.size):
            region1 = np.bitwise_or(region1, region1_[:, i])

        region = region1 * region2

        plot_safety_region(ax3, xs, region)
        ax3.set_yticks([])
        ax3.set_ylim(bottom=0.03, top=1)

        if signal_v is not None:
            ax.axvline(x=signal_v, zorder=4, color='black')
            ax2.axvline(x=signal_v, zorder=4, color='black')
            ax3.axvline(x=signal_v, zorder=4, color='black')
            ax3.scatter([signal_v], [0.5], zorder=6, color='black', marker='d')

        if self.mode == 'both':
            x_pos = .6*float(xs[-1])
            self.right_axes[1].text(x_pos, .37, f"τ = {self.tau:.2f}")
            self.right_axes[1].text(x_pos, .3, f"z = {self.z:.2f}")
            self.right_axes[1].text(x_pos, .23, f"signal e = {signal_v:.2f}")


# The iterable type is incorrect, but it's the closest thing we can get. The object should
# support indexadability (which lists and ndarrays do support)
def completing_polygon(xs: Iterable[float], ys: Iterable[float]) -> Any:
    """
    xs: is a list of accending numbers
    ys: is either 0 or 1
    """
    # Extending: [(xs[0], 0), *zip(xs, region), (xs[-1], 0)]
    x_, y_ = xs[0], ys[0]  # type: ignore
    yield x_, 0

    for x, y in zip(xs, ys):
        if y != y_:
            if y == 0:
                yield x_, y
            else:
                yield x, y_
        yield x, y
        x_, y_ = x, y

    yield xs[-1], 0  # type: ignore


def plot_safety_region(axis: Any, xs: Iterable[float], region: Iterable[float]) -> None:
    # region_verts = [(xs[0], 0), *zip(xs, region), (xs[-1], 0)]
    region_verts = list(completing_polygon(xs, region))
    region_poly = Polygon(region_verts, facecolor='#60af3d', edgecolor='#fff', zorder=2)
    axis.add_patch(region_poly)


def animate(
    seplot: SEPlot,
    input_data: Iterable[Tuple[Any, Any, Any]],
    size: Optional[int] = None,
    save_as: Optional[str] = None,
    frames: int = 10
) -> None:
    ani = animation.FuncAnimation(
        seplot.fig, seplot.fig_plot, input_data, blit=False,
        interval=frames, repeat=False, save_count=size)  # , init_func=init_func)

    if save_as is not None:
        # ani.save(f"{save_as}.gif", animation.ImageMagickWriter(fps=20))
        ani.save(f"{save_as}.m4v", animation.FFMpegWriter(fps=20))
    else:
        plt.show()


if __name__ == '__main__':
    data.legacy_data(revert_to_bad=True)

    z = 4
    tau = .99

    # Don't delete. This plot has been "finished"
    # seplot = SEPlot(z, tau, undertitle=True, right_limit=18000, mode='right')
    # animate(seplot, np.linspace(6, 17, 900), save_as='plots/plot_007-right')

    # angles_attack = np.linspace(1, 16, 1000)
    # airspeed = 8.7 * np.ones(angles_attack.shape)
    # signal = data.synthetize_signal(airspeed, angles_attack)
    # seplot = SEPlot(z, tau, undertitle=True, right_limit=18000, mode='both',
    #                 tick_time=1, time_width=300)
    # animate(seplot, zip(airspeed, angles_attack, signal), size=len(airspeed),
    #         save_as="plots/plot_007-both-variable-AoA_constant-airspeed", frames=20)

    airspeed = np.linspace(17, 6, 1000)
    angles_attack = 11.5 * np.ones(airspeed.shape)
    signal = data.synthetize_signal(airspeed, angles_attack)
    seplot = SEPlot(z, tau, undertitle=True, right_limit=18000, mode='both',
                    tick_time=1, time_width=300)
    animate(seplot, zip(airspeed, angles_attack, signal), size=len(airspeed),
            save_as="plots/plot_007-both-constant-AoA_variable_airspeed", frames=20)
    # seplot.fig_plot(next(zip(airspeed, angles_attack, signal)))

    # airspeed = 13.3
    # seplot.fig_plot(airspeed)
    # plt.show()
