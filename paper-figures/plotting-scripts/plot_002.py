# import plotly.graph_objects as go
import numpy as np
import scipy.io as sio

# np.random.seed(1)

# N = 100
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# sz = np.random.rand(N) * 30
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=x,
#     y=y,
#     mode="markers",
#     marker=go.scatter.Marker(
#         size=sz,
#         color=colors,
#         opacity=0.6,
#         colorscale="Viridis"
#     )
# ))
# fig.add_trace(
#     go.Bar(x=(0,), y=(0,))
# )

import plotly.express as px
import pandas as pd
# df = px.data.iris()
# fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="rug",
# marginal_x="histogram")

# fig.show()

mat_contents = sio.loadmat('windTunnel_data_sensor3_AS15.mat')

seTW = mat_contents['seTW']
final_shape = np.ones((91, 8, 18, 15), dtype=np.int)  # np.ones(seTW.shape)
sensors = np.arange(1, 9).reshape((1, 8, 1, 1)) * final_shape
aoas = np.arange(1, 19).reshape((1, 1, 18, 1)) * final_shape
airspeeds = np.array(
    [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
).reshape((1, 1, 1, 15)) * final_shape

seTWdf = pd.DataFrame({
    'signal': seTW.flatten(),
    'sensor': sensors.flatten(),
    'AoA': aoas.flatten(),
    'airspeed': airspeeds.flatten()
})

df = seTWdf[seTWdf['sensor'] == 3][seTWdf['airspeed'] == 19]

fig = px.box(df, y="signal", x="AoA")
# fig = px.violin(df, y="signal", x="AoA")  # , box=True)

fig.show()
# np.dtype([("signal", np.float64), ("sensor", np.int), ("AoA", np.int), ("airspeed", pn.int)])
