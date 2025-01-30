import numpy as np
from scipy.optimize import fsolve

from astropy import constants, units

def kepler_eq(E, e, M):
    return E-e*np.sin(E)-M

def eccentric_anomaly(e, M, tol = 1e-15):
    return fsolve(kepler_eq, np.pi, args = (e, M), xtol = tol)[0]