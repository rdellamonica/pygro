import numpy as np
import time
import sympy as sp
import pygro.integrators as integrators
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import scipy.interpolate as sp_int

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


class GeodesicEngine():

    def __init__(self, metric, verbose = True, integrator = "dp45"):

        self.link_metrics(metric, verbose)
        self.integrator = integrator
        self.wrapper = None
    
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

        if len(self.metric.get_parameters_functions()) > 0:
            self.wrapper = "lambdify"
        else:
            self.wrapper = "autowrap"

        self.motion_eq_f = []

        for eq in [*self.eq_x, *self.eq_u]:
            if self.wrapper == "autowrap":
                self.motion_eq_f.append(autowrap(self.metric.subs_functions(eq), backend='cython', args = [*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()]))
            else:
                self.motion_eq_f.append(lambdify([*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()], self.metric.subs_functions(eq)))

        def f_eq(tau, xu):
            return np.array([self.motion_eq_f[i](*xu, *self.metric.get_parameters_val()) for i in range(8)])

        self.motion_eq = f_eq
        
        u0_f_timelike = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_timelike))
        
        def f_u0_timelike(initial_x, u1, u2, u3):
            return abs(u0_f_timelike(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))

        u0_f_null = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_null))

        def f_u0_null(initial_x, u1, u2, u3):
            return abs(u0_f_null(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))

        self.u0_f_timelike = f_u0_timelike
        self.u0_f_null = f_u0_null
        
        if verbose:
            print("Metric linking complete.")

    def stopping_criterion(self, *x):
        return True

    def set_stopping_criterion(self, expr):
        expr_s = sp.parse_expr(expr)
        free_symbols = list(expr_s.free_symbols-set(self.metric.x))
        check = True
        for symbol in free_symbols:
            if symbol in self.metric.get_parameters_symb():
                check &= True
        if check == True:
            stopping_criterion = sp.lambdify([*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()], expr_s)
            def f(*xu):
                return stopping_criterion(*xu, *self.metric.get_parameters_val())

            self.stopping_criterion = f
        else:
            raise TypeError("Unkwnown symbol {}".format(str(symbol)))
        

    def integrate(self, geo, tauf, initial_step, verbose=False, direction = "fw", interpolate = False, **params):
        if verbose:
            print("Integrating...")

        integrator = integrators.get_integrator(self.integrator, self.motion_eq, stopping_criterion = self.stopping_criterion, verbose = verbose, **params)

        if direction == "bw":
            h = -initial_step
            tauf = -tauf
        else:
            h = initial_step

        if verbose:
            time_start = time.perf_counter()

        tau, xu = integrator.integrate(0, tauf, np.array([*geo.initial_x, *geo.initial_u]), h)
                
        if verbose:
            time_elapsed = (time.perf_counter() - time_start)
            print("Integration time = {} s".format(time_elapsed))

        geo.tau = tau
        geo.x = np.stack(xu[:,:4])
        geo.u = np.stack(xu[:,4:])

        if interpolate == True:
            geo.x_int = sp_int.interp1d(geo.tau, geo.x, axis = 0, kind = "cubic")
            geo.u_int = sp_int.interp1d(geo.tau, geo.u, axis = 0, kind = "cubic")        
    
    def set_integrator(self, integrator):
        self.integrator = integrator