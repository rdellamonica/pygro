import pygro
import numpy as np

name = "Schwarzschild spacetime"
coordinates = ["t", "r", "theta", "phi"]

transform_functions = [
    "t",
    "r*sin(theta)*cos(phi)",
    "r*sin(theta)*sin(phi)",
    "r*cos(theta)"
]

line_element = "-A(r)*dt**2+1/A(r)*dr**2+(r**2)*(dtheta**2+B(theta)**2*dphi**2)"

A = "1-2*M/r"
B = "sin(theta)"


"""
def A(t, r, theta, phi):
    M = metric.get_constant("M")
    return 1-2*M/r

def dAdr(t, r, theta, phi):
    M = metric.get_constant("M")
    return 2*M/r**2
"""

metric = pygro.Metric(
    name = name,
    coordinates = coordinates,
    line_element = line_element,
    transform = transform_functions,
    A = A,
    B = B,
    M = 1,
)

geo_engine = pygro.GeodesicEngine(backend="lambdify")
geo_engine.set_stopping_criterion("r > 2.00001*M", "horizon")

geo = pygro.Geodesic("null", geo_engine, verbose = True)
geo.set_starting_point(0, 50, np.pi/2, 0)
geo.set_starting_4velocity(u1 = -1, u2 = 0, u3 = 0.001)
geo_engine.integrate(geo, 5000000, 1, verbose=True, AccuracyGoal = 16, PrecisionGoal = 16)