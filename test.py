import pigro
import numpy as np

metric = pigro.metric()
metric.load_metric("pigro/default_metrics/kerr_BL.metric", m = 1, a = 0.95)

geo_engine = pigro.geodesic_engine(metric)

geo = pigro.geodesic("null", geo_engine)

geo.set_starting_point(0, 10, np.pi/2, 0)
geo.set_starting_velocity_direction(10, 0, angles = "deg")