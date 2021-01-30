import numpy as np
import time
from sympy import *

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
        self.eq_x = []
        self.eq_u = []

        ################################################################
        #   Calculating equations of motion based                      #
        #   on the christhoffel symbols of the metric                  #
        ################################################################
        if verbose:
            print("Calculating symbolic equations of motion:")

        for rho in range(4):
            if verbose:
                print("- {}/4".format(rho+1))
            eq = 0
            for mu in range(4):
                for nu in range(4):
                    eq += -g.chr(mu, nu, rho)*g.u[mu]*g.u[nu]
            self.eq_u.append(eq)
            self.eq_x.append(self.metric.u[rho])

        ################################################################
        #   Adding to class a symbolic method to retrieve u_0          #
        #   starting from the other components of u                    #
        ################################################################

        if verbose:
            print("Adding to class a method to get initial u_0 and u_1...")

        eq = 0

        for mu in range(4):
            for nu in range(4):
                eq += self.metric.g[mu, nu]*self.metric.u[mu]*self.metric.u[nu]

        self.u0_s_null = solve(eq, self.metric.u[0], simplify=False, rational=False)
        self.u1_s_null = solve(eq, self.metric.u[1], simplify=False, rational=False)

        eq = 1

        for mu in range(4):
            for nu in range(4):
                eq += self.metric.g[mu, nu]*self.metric.u[mu]*self.metric.u[nu]

        self.u0_s_timelike = solve(eq, self.metric.u[0], simplify=False, rational=False)
        self.u1_s_timelike = solve(eq, self.metric.u[1], simplify=False, rational=False)

        if verbose:
            print("OK")

        if verbose:
            print("Metric linking complete.")

    def stopping_criterion(self, x):
        return True

    def set_stopping_criterion(self, f):
        self.stopping_criterion = f

    def integrate(self, geo, tauf, show_time = False, eval_constants = False, h = 0.01, delta = 1e-10, verbose = True, direction = "fw"):
        if verbose:
            print("Integrating...")
    
        geo.tau.append(0)

        if direction == "bw":
            h = -h
            tauf = -tauf

        geo.x = []
        geo.u = []

        if eval_constants:
            geo.constants = [self.eval_constants_of_motion(geo.initial_x, geo.initial_u)]
        
        geo.x.append(geo.initial_x)
        geo.u.append(geo.initial_u)

        if show_time:
            time_start = time.perf_counter()

        while abs(geo.tau[-1]) < abs(tauf) and self.stopping_criterion(geo):
            next = self.ck4(geo.x[-1], geo.u[-1], geo.tau[-1], h, delta)
            if eval_constants:
                geo.constants.append(self.eval_constants_of_motion(next[0], next[1]))
            geo.x.append(next[0])
            geo.u.append(next[1])
            geo.tau.append(next[2])
            h = next[3]
            if show_time:
                print("Tau = {}".format(next[2]), end = "\r")
                
        if show_time:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        if eval_constants:
                geo.constants = np.stack(geo.constants)

        geo.x = np.stack(geo.x)
        geo.u = np.stack(geo.u)
    
    def integrate2(self, geo, tauf, show_time = False, eval_constants = False, h = 0.01, Atol = 1e-10, Rtol = 1e-5, verbose = True, direction = "fw", precision = 10):
        if verbose:
            print("Integrating...")
    
        geo.tau.append(0)

        if direction == "bw":
            h = -h
            tauf = -tauf

        geo.x = []
        geo.u = []

        if eval_constants:
            geo.constants = [self.eval_constants_of_motion(geo.initial_x, geo.initial_u)]
        
        geo.x.append(geo.initial_x)
        geo.u.append(geo.initial_u)

        if show_time:
            time_start = time.perf_counter()

        while abs(geo.tau[-1]) < abs(tauf) and self.stopping_criterion(geo):
            next = self.dp4(geo.x[-1], geo.u[-1], geo.tau[-1], h, Atol, Rtol)
            if eval_constants:
                geo.constants.append(self.eval_constants_of_motion(next[0], next[1]))
            geo.x.append(next[0])
            geo.u.append(next[1])
            geo.tau.append(next[2])
            h = next[3]
            if show_time:
                print("Tau = {}".format(next[2]), end = "\r")
                
        if show_time:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        if eval_constants:
                geo.constants = np.stack(geo.constants)

        geo.x = np.stack(geo.x)
        geo.u = np.stack(geo.u)
    
    def integrate3(self, geo, tauf, show_time = False, eval_constants = False, h = 0.01, delta = 1e-10, verbose = True, direction = "fw"):
        if verbose:
            print("Integrating...")
    
        geo.tau.append(0)

        if direction == "fw":
            geo.initial_u[0] = abs(geo.initial_u[0])
        elif direction == "bw":
            geo.initial_u[0] = -abs(geo.initial_u[0])

        geo.x = []
        geo.u = []

        if eval_constants:
            geo.constants = [self.eval_constants_of_motion(geo.initial_x, geo.initial_u)]
        
        geo.x.append(geo.initial_x)
        geo.u.append(geo.initial_u)

        if show_time:
            time_start = time.perf_counter()

        while geo.tau[-1] < tauf and self.stopping_criterion(geo):
            next = self.bs23(geo.x[-1], geo.u[-1], geo.tau[-1], h, delta)
            if eval_constants:
                geo.constants.append(self.eval_constants_of_motion(next[0], next[1]))
            geo.x.append(next[0])
            geo.u.append(next[1])
            geo.tau.append(next[2])
            h = next[3]
            if show_time:
                print("Tau = {}".format(next[2]), end = "\r")
                
        if show_time:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        if eval_constants:
                geo.constants = np.stack(geo.constants)

        geo.x = np.stack(geo.x)
        geo.u = np.stack(geo.u)
    
    def integrate4(self, geo, tauf, show_time = False, eval_constants = False, h = 0.01, delta = 1e-10, verbose = True, direction = "fw"):
        if verbose:
            print("Integrating...")

        if direction == "bw":
            h = -h
            tauf = -tauf

        geo.tau = []
        geo.x = []
        geo.u = []

        geo.tau.append(0)

        if eval_constants:
            geo.constants = [self.eval_constants_of_motion(geo.initial_x, geo.initial_u)]
        
        geo.x.append(geo.initial_x)
        geo.u.append(geo.initial_u)

        if show_time:
            time_start = time.perf_counter()

        while geo.tau[-1] < tauf and self.stopping_criterion(geo):
            next = self.dp78(geo.x[-1], geo.u[-1], geo.tau[-1], h, delta)
            if eval_constants:
                geo.constants.append(self.eval_constants_of_motion(next[0], next[1]))
            geo.x.append(next[0])
            geo.u.append(next[1])
            geo.tau.append(next[2])
            h = next[3]
            if show_time:
                print("Tau = {}".format(next[2]), end = "\r")
                
        if show_time:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        if eval_constants:
                geo.constants = np.stack(geo.constants)

        geo.x = np.stack(geo.x)
        geo.u = np.stack(geo.u)

    def eval_constants_of_motion(self, x, u):
        c = []
        for constant in self.metric.constants_of_motion:
            c.append(self.metric.constants_of_motion[constant](x, u))

        return c
    
    def set_constants(self):

        motion_eq_f = lambdify([g.x, g.u], [self.eq_x, self.eq_u], 'numpy')

        def f(x, u):
            return np.array(motion_eq_f(x, u))
        
        self.motion_eq = f

        eq_x = self.eq_x_s
    
        subs = []
        for param in params:
            x = param['symbol']
            y = param['value']
            subs.append([x,y])
        
        eq_u = []
        for i in range(4):
            eq_u.append(self.eq_u_s[i].subs(subs))
        
        u0 = self.u0_f_null_s.subs(subs)
        self.u0_f_null = lambdify([self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3]], u0, 'numpy')
        u0 = self.u0_f_timelike_s.subs(subs)
        self.u0_f_timelike= lambdify([self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3]], u0, 'numpy')

        motion_eq_f = lambdify([self.metric.x, self.metric.u], [eq_x, eq_u], 'numpy')

        def f(x, u):
            return np.array(motion_eq_f(x, u))

        self.motion_eq = f