import pigro
import numpy as np
import matplotlib.pyplot as plt

metric = pigro.metric()
metric.load_metric("pigro/default_metrics/Kerr-BL.metric", m = 1, a = 0.95)

geo_engine = pigro.geodesic_engine(metric)

geo = pigro.geodesic("null", geo_engine)

geo.set_starting_point(0, 50, np.pi/2, 0)
geo.set_starting_velocity_direction(0, 0, angles = "deg")

geo_engine.integrate(geo, 48.1, initial_step = 1)

x = geo.x[:,1]*np.cos(geo.x[:,3])
y = geo.x[:,1]*np.sin(geo.x[:,3])

fig, ax = plt.subplots()

ax.axis('equal')

ax.plot(x, y)

plt.show()