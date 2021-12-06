from __future__ import annotations

import matplotlib.pyplot as plt

import data_104 as data_experiments
from misc import Plot

dirname = "plots/plot_104"


if False and __name__ == '__main__':
    airspeed = 10
    sensor1 = 0  # Numbering starting at zero
    sensor2 = 6

    # Things to show:
    # 1. Close sensors: (0, 1), (0, 2), (0, 4)
    # 2. Far away sensors: (0, 6) and (0, 7)
    # 3. Sensor 6 is different enough from all the others to let us separate
    #    stall from no stall quite easily when paired with other sensor. Probably
    #    the best classification will be found if we use sensors 0, 6 and 7

    airspeed_i = data_experiments.airspeed_index(airspeed)
    assert isinstance(airspeed_i, int)
    total, means, vars_, stall = \
        data_experiments.dists_given_airspeed(airspeed, (sensor1, sensor2))

    # Shows me some correlations
    if False:  # correlation between signals MEANS
        fig, ax = plt.subplots(1, 1)
        # Plotting energy signal means (no-stall)
        ax.scatter(means[stall == 0, 0], means[stall == 0, 1], marker='.')
        # Plotting energy signal means (no-stall)
        ax.scatter(means[stall == 1, 0], means[stall == 1, 1], marker='.')
        ax.set_title(f"Mean signal energies for Airspeed = {airspeed}\n"
                     f"Sensors: {sensor1+1} and {sensor2+1}")
        plt.show()

    # Shoot. They are actually highly correlated.
    # Close sensors are highly correlated. Far away sensors are not much
    if True:  # correlation between raw signals
        # with Plot(dirname, f"correlation-airspeed={airspeed}-sensors={sensor1+1}n{sensor2+1}"):
        with Plot():
            fig, ax = plt.subplots(1, 1)
            # Plotting energy signals for no-stall for both sensors
            ax.scatter(data_experiments.seTW[:, sensor1, stall == 0, airspeed_i],
                       data_experiments.seTW[:, sensor2, stall == 0, airspeed_i],
                       marker='.',
                       label="No-stall")
            # Plotting energy signals for stall for both sensors
            ax.scatter(data_experiments.seTW[:, sensor1, stall == 1, airspeed_i],
                       data_experiments.seTW[:, sensor2, stall == 1, airspeed_i],
                       marker='x',
                       label="Stall")
            ax.set_title(f"All signal energies for Airspeed = {airspeed}\n"
                         f"Sensors: {sensor1+1} and {sensor2+1}")
            ax.set_xlabel(f"Signal energy for sensor {sensor1+1}")
            ax.set_ylabel(f"Signal energy for sensor {sensor2+1}")
            plt.legend()

    # Highly correlated for close sensors and lower correlation for far away sensors
    if False:  # correlation between raw signals for all configurations between two sensors
        fig, ax = plt.subplots(1, 1)
        # Plotting energy signals for both sensors
        ax.scatter(data_experiments.seTW[:, sensor1, :, :],
                   data_experiments.seTW[:, sensor2, :, :], marker='.')
        ax.set_title(f"All signal energies for all configurations\n"
                     f"Sensors: {sensor1+1} and {sensor2+1}")
        plt.show()

    # This shows that albeit the signals are highly correlated, they differ in magnitude
    if False:  # printing signal energies
        fig, ax = plt.subplots(1, 1)
        AoA = 14
        ax.plot(data_experiments.seTW[:, sensor1, AoA, airspeed_i],
                label=f'Sensor {sensor1+1}')
        ax.plot(data_experiments.seTW[:, sensor2, AoA, airspeed_i],
                label=f'Sensor {sensor2+1}')
        ax.set_title(f"Signal energies for AoA = {AoA} Airspeed = {airspeed}")
        plt.legend()
        plt.show()

    if False:  # printing all signal energies for both sensors
        fig, (ax1, ax2) = plt.subplots(2, 1)
        # Plotting energy signals for no-stall for both sensors
        ax1.plot(data_experiments.seTW[:, sensor1, :, airspeed_i].T.flatten())
        ax1.set_ylabel(f"Sensor {sensor1+1}")
        ax2.plot(data_experiments.seTW[:, sensor2, :, airspeed_i].T.flatten())
        ax2.set_ylabel(f"Sensor {sensor2+1}")
        ax1.set_title(f"All signal energies for Airspeed = {airspeed}")
        plt.show()
