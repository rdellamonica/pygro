import numpy as np
import time
from sympy import *
import pygro.integrators as integrators

#########################################################################################
#                               GEODESIC ENGINE                                         #
#                                                                                       #
#   This is the main engine for numerical integration of the geodesic equation.         #
#   Two classes are defined:                                                            #
#    - The integration engine itself, which is linked to a metric object and            #
#      retrieves the equation of motion symbolically.                                   #
#    - The geodesic object, in which the numerical integration result is stored         #
#                                                                                       #
#########################################################################################


################################################################
#   The geodesic_engine object needs as argument               #
#    a metric object which is linked to the metric             #
################################################################


class geodesic_engine(object):

    def __init__(self, metric, verbose = True):

        self.link_metrics(metric, verbose)
    
    def link_metrics(self, g, verbose):
        if verbose:
            print("Linking {} to the Geodesic Engine".format(g.name))

        self.metric = g
        self.eq_x = g.eq_x
        self.eq_u = g.eq_u

        self.u0_s_null = g.u0_s_null
        self.u0_s_timelike = g.u0_s_timelike

        self.metric.geodesic_engine_linked = True
        self.metric.geodesic_engine = self

        self.evaluate_constants()

        if verbose:
            print("Metric linking complete.")

    def stopping_criterion(self, x):
        return True

    def set_stopping_criterion(self, f):
        self.stopping_criterion = f

    def integrate(self, geo, tauf, initial_step = 0.01, Atol = 0, Rtol = 1e-6, verbose = True, direction = "fw", method = "dp4"):
        if verbose:
            print("Integrating...")

        try:
            integrator = integrators.__dict__[method]
        except:
            print(f"Method {method} not found. Currently supported methods: ck4, dp4, fh78, dp78, bs23, rk45, rkf45.")

        if direction == "bw":
            h = -h
            tauf = -tauf

        geo.tau = [0]
        geo.x = [] 
        geo.u = []
        
        geo.x.append(geo.initial_x)
        geo.u.append(geo.initial_u)

        if verbose:
            time_start = time.perf_counter()

        h = initial_step

        while abs(geo.tau[-1]) < abs(tauf) and self.stopping_criterion(geo):
            next = integrator(self, geo.x[-1], geo.u[-1], geo.tau[-1], h, Atol, Rtol)
            geo.x.append(next[0])
            geo.u.append(next[1])
            geo.tau.append(next[2])
            h = next[3]
            if verbose:
                print("Tau = {}".format(next[2]), end = "\r")
                
        if verbose:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        geo.x = np.stack(geo.x)
        geo.u = np.stack(geo.u)
    
    def evaluate_constants(self):

        eq_u = []
        for eq in self.eq_u:
            eq_u.append(self.metric.evaluate_constants(eq))

        motion_eq_f = lambdify([self.metric.x, self.metric.u], [self.eq_x, eq_u], 'numpy')

        def f(x, u):
            return np.array(motion_eq_f(x, u))

        self.motion_eq = f
        
        u0_timelike = self.metric.evaluate_constants(self.u0_s_timelike)
        u0_null = self.metric.evaluate_constants(self.u0_s_null)

        self.u0_f_null = lambdify([self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3]], u0_null, 'numpy')
        self.u0_f_timelike = lambdify([self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3]], u0_timelike, 'numpy')