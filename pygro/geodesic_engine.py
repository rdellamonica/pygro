import numpy as np
import numpy.typing as npt
import time
import sympy as sp
import pygro.integrators as integrators
from pygro.metric_engine import Metric
from pygro.geodesic import Geodesic
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import scipy.interpolate as sp_int
import logging
from typing import Optional, Literal, Union, Callable

_BACKEND_TYPES = Literal["autowrap", "lambdify"]

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
    instances = []
    
    def __init__(self, metric: Optional[Metric] = None, backend: _BACKEND_TYPES = "autowrap", integrator = "dp45"):
        if metric == None:
            if isinstance(Metric.instances[-1], Metric):
                metric = Metric.instances[-1]
                logging.warning(f"No Metric object passed to the Geodesic Engine constructor. Using last initialized Metric ({metric.name}) instead")
                
        self.link_metric(metric, backend)
        self.integrator = integrator
        GeodesicEngine.instances.append(self)
    
    def link_metric(self, g: Metric, backend: _BACKEND_TYPES):
        logging.info(f"Linking {g.name} to the Geodesic Engine")

        self.metric = g
        self.eq_x = g.eq_x
        self.eq_u = g.eq_u
        
        self.u0_s_null = g.u0_s_null
        self.u0_s_timelike = g.u0_s_timelike

        self.metric.geodesic_engine_linked = True
        self.metric.geodesic_engine = self

        if (backend == "lambdify") or len(self.metric.get_parameters_functions()) > 0:
            self._wrapper = "lambdify"
        else:
            self._wrapper = "autowrap"

        self._motion_eq_f = []

        for eq in [*self.eq_x, *self.eq_u]:
            if self._wrapper == "autowrap":
                self._motion_eq_f.append(autowrap(self.metric.subs_functions(eq), backend='cython', args = [*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()]))
            else:
                self._motion_eq_f.append(lambdify([*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()], self.metric.subs_functions(eq)))

        def _f_eq(tau, xu):
            return np.array([self._motion_eq_f[i](*xu, *self.metric.get_parameters_val()) for i in range(8)])

        self.motion_eq = _f_eq
        
        u0_f_timelike = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_timelike))
        
        def f_u0_timelike(initial_x, u1, u2, u3):
            return abs(u0_f_timelike(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))

        u0_f_null = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_null))

        def f_u0_null(initial_x, u1, u2, u3):
            return abs(u0_f_null(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))

        self.u0_f_timelike = f_u0_timelike
        self.u0_f_null = f_u0_null
        
        self.stopping_criterion = StoppingCriterionList([])
        
        logging.info("Metric linking complete.")

    def set_stopping_criterion(self, expr: str, exit_str: str = "none"):
        self.stopping_criterion = StoppingCriterionList([])
        self.add_stopping_criterion(expr, exit_str)
        
    def add_stopping_criterion(self, expr: str, exit_str: str = "none"):
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

            self.stopping_criterion.append(StoppingCriterion(f, expr, exit_str))
        else:
            raise TypeError(f"Unkwnown symbol {str(symbol)}")
        

    def integrate(self, geo: Geodesic, tauf: float, initial_step: float, verbose: bool = False, direction: Literal["fw", "bw"] = "fw", interpolate: bool = False, **integrator_kwargs):
        if verbose:
            logging.info("Integrating...")

        integrator = integrators.get_integrator(self.integrator, self.motion_eq, stopping_criterion = self.stopping_criterion, verbose = verbose, **integrator_kwargs)

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
            logging.info(f"Integration completed in {time_elapsed:.5} s with result '{exit}'")

        geo.tau = tau
        geo.x = np.stack(xu[:,:4])
        geo.u = np.stack(xu[:,4:])
        geo.exit = exit     
    
    def set_integrator(self, integrator):
        self.integrator = integrator

class StoppingCriterion:
    def __init__(self, stopping_criterion: Callable[[npt.ArrayLike], bool], expr: str, exit_str: str):
        self.stopping_criterion = stopping_criterion
        self.exit = exit_str
        self.expr = expr
        
    def __repr__(self):
        return f"<StoppingCriterion: {self.expr}>"
    
    def __call__(self, *geo: npt.ArrayLike) -> bool:
        return self.stopping_criterion(*geo)

class StoppingCriterionList:
    def __init__(self, stopping_criterions: list[StoppingCriterion]):
        self.stopping_criterions = stopping_criterions
        self.exit = None
        
    def append(self, stopping_criterion: StoppingCriterion):
        self.stopping_criterions.append(stopping_criterion)
        
    def __repr__(self):
        return f"<StoppingCriterions: [{', '.join([f'{s_c.expr}' for s_c in self.stopping_criterions])}]>"
        
    def __call__(self, *geo: npt.ArrayLike) -> bool:
        check = True
        for stopping_criterion in self.stopping_criterions:
            check &= stopping_criterion(*geo)
            if not check:
                self.exit = stopping_criterion.exit

        return check