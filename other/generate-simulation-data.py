import params
import numpy as np
from numpy.random import normal
import sys
import ast

np.random.seed(4235243)


def generate_data(airspeed_i: int, aoa_i: int, noise: float) -> np.ndarray:
    if noise < 0:
        raise Exception("`noise` should be positive")

    data = normal(loc=params.means[airspeed_i][aoa_i],
                  scale=params.stds[airspeed_i][aoa_i],
                  size=100)

    data_error = normal(loc=params.means[airspeed_i][aoa_i],
                        scale=params.stds[airspeed_i][aoa_i]*(1 + noise),
                        size=100)

    data_error[data_error < 0] = 0

    return np.concatenate([data, data_error])


if __name__ == '__main__':
    airspeed_i = 0
    aoa_i = 5
    # noise = 6.0

    if len(sys.argv) > 1:
        noise = ast.literal_eval(sys.argv[1])
        if not isinstance(noise, (float, int)):
            print("`noise` must be a number")
            exit(1)
    else:
        print("You must supply a number argument")
        exit(1)

    data = generate_data(airspeed_i, aoa_i, noise)

    for i in data:
        print(i)
