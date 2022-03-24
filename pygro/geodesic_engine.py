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
    r"""This is the main symbolic tool within PyGRO to perform tensorial calculations
    starting from the spacetime metric. The ``Metric`` object can be initialized in two separate ways:

    .. rubric:: Interactive mode
    
    In this case the ``Metric`` has to be initialized without any additional argument.
    Later the :py:func:`Metric.initialize_metric` method should be called without additional arguments in order to enter the interactive mode.
    During this phase the user will be asked to input:

    * The ``name`` of the spacetime.
    * The symbolic ``coordinates`` in which the metric is expressed.
    * The symbolic expression of the spacetime metric which can be either expressed as a line elemnt :math:`ds^2 = g_{\mu\nu}dx^\mu dx^\nu` or as the single components :math:`g_{\mu\nu}` of the metric tensor.
    * The symbolix expression of ``transformation functions`` to pseudo-cartesian coordinates which can be useful to :doc:`visualize`.

    After the insertion of all the required information the metric is initialized.

    .. rubric:: Functional mode

    In this case the ``Metric`` object is initialized without interactive input by the user. The arguments to build the metric can be either passed to the ``Metric`` constructor upon class initialization,
    
    >>> spacetime_metric = pygro.Metric(**kwargs)

    or as arguments to the  :py:func:`Metric.initialize_metric` method:

    >>> spacetime_metric = pygro.Metric()
    >>> spacetime_metric.initialize_metric(**kwargs)

    The ``**kwargs`` that can be passed to initialize a ``Metric`` object are the following:

    :param name: The name of the metric to initialize.
    :type name: str
    :param coordinates: Four-dimensional list containing the symbolic expression of the space-time coordinates in which the `line_element` argument is written.
    :type coordinates: list of str
    :param line_element: A string containing the symbolic expression of the line element (that will be parsed using `sympy`) expressed in the space-time coordinates defined in the `coordinates` argument.
    :type line_element: str

    .. note::
        Test
    
    After successful initialization, the ``Metric`` instance has the following attributes:

    :ivar g: The symbolic representation of the metric tensor. It is a :math:`4\times4` ``sympy.Matrix`` object.
    :type g: sympy.Matric

    """
    def __init__(self, metric, verbose = True, backend = "autowrap", integrator = "dp45"):

        self.link_metrics(metric, verbose, backend)
        self.integrator = integrator
        self.wrapper = None
    
    def link_metrics(self, g, verbose = "True", backend = "autowrap"):
        if verbose:
            print("Linking {} to the Geodesic Engine".format(g.name))

        self.metric = g
        self.eq_x = g.eq_x
        self.eq_u = g.eq_u
        
        self.u0_s_null = g.u0_s_null
        self.u0_s_timelike = g.u0_s_timelike

        self.metric.geodesic_engine_linked = True
        self.metric.geodesic_engine = self

        if (backend == "lambdify") or len(self.metric.get_parameters_functions()) > 0:
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

    def set_stopping_criterion(self, expr, exit_str = "none"):
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

            self.stopping_criterion = StoppingCriterion(f, exit_str)
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

        tau, xu, exit = integrator.integrate(0, tauf, np.array([*geo.initial_x, *geo.initial_u]), h)
                
        if verbose:
            time_elapsed = (time.perf_counter() - time_start)
            print(f"Integration completed in {time_elapsed:.5} s with result '{exit}'")

        geo.tau = tau
        geo.x = np.stack(xu[:,:4])
        geo.u = np.stack(xu[:,4:])
        geo.exit = exit

        if interpolate == True:
            geo.x_int = sp_int.interp1d(geo.tau, geo.x, axis = 0, kind = "cubic")
            geo.u_int = sp_int.interp1d(geo.tau, geo.u, axis = 0, kind = "cubic")        
    
    def set_integrator(self, integrator):
        self.integrator = integrator
    

class StoppingCriterion:

    def __init__(self, stopping_criterion, exit_str):
        self.stopping_criterion = stopping_criterion
        self.exit = exit_str
    
    def __call__(self, *geo):
        return self.stopping_criterion(*geo)
