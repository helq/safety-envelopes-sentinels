import params
import numpy as np
from numpy.random import normal

airspeed_i = 0
aoa_i = 5

data = normal(loc=params.means[airspeed_i][aoa_i],
              scale=params.stds[airspeed_i][aoa_i],
              size=100)

data_error = normal(loc=params.means[airspeed_i][aoa_i],
                    scale=params.stds[airspeed_i][aoa_i]*6,
                    size=100)

data_error[data_error < 0] = 0

for i in np.concatenate([data, data_error]):
    print(i)
