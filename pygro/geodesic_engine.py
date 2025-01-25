from typing import Optional, Literal, Callable, TYPE_CHECKING, get_args
import numpy as np
import numpy.typing as npt
import time
import sympy as sp
import pygro.integrators as integrators
from pygro.integrators import AVAILABLE_INTEGRATORS
from pygro.metric_engine import Metric
if TYPE_CHECKING:
    from pygro.geodesic import Geodesic
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify
import logging

_BACKEND_TYPES = Literal["autowrap", "lambdify"]

class GeodesicEngine():
    r"""
        The main numerical class in PyGRO. It deals with performing the numerical operations to integrate a :py:class:`.Geodesic` object. 
        
        After linking a :py:class:`.Metric` to the :py:class:`GeodesicEngine`, the latter will have the following attributes:
        
        :ivar motion_eq: A callable of the coordinates and their derivatives which returns the right-hand-side of the geodesic equations (casted to a first-order system of ODEs). For example, given the coordinates ``["t", "x", "y", "z"]``, the ``motion_eq`` will be of type ``(t, x, y, z, u_t, u_x, u_y, u_z) -> Iterable[float]`` of dimension 8. 
        :vartype motion_eq: Iterable[Callable]
        :ivar u[i]_f_timelike: a helper function that returns the numerical value of the *i*-th component of the 4-velocity of a massive test particle as a function of the others, which normalizes the 4-velocity to :math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = -1`
        :vartype u[i]_f_timelike: Callable[Iterable[float], float]
        :ivar u[i]_f_null: a helper function that returns the numerical value of the *i*-th component of the 4-velocity of a null test particle as a function of the others, which normalizes the 4-velocity to :math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0`
        :vartype u[i]_f_null: Callable[Iterable[float], float]
        :ivar stopping_criterion: a :py:class:`.StoppingCriterionList` representing the list of conditions to check at each integration step to determine whether to stop the integration or not.
        :vartype stopping_criterion: StoppingCriterionList
    """
    instances = []
    
    def __init__(self, metric: Optional[Metric] = None, backend: _BACKEND_TYPES = "autowrap", integrator : AVAILABLE_INTEGRATORS = "rkf45"):
        r"""
            The :py:class:`GeodesicEngine` constructor accepts the following arguments:
            
            :param metric: The :py:class:`.Metric` object to link to the :py:class:`GeodesicEngine` and from which the geodesic equations and all the related symbolic quantities are retrieved. If not provided, the :py:class:`GeodesicEngine` will be linked to the last initialized :py:class:`.Metric`.
            :type metric: Metric
            :param backend: The symbolic backend to use. If ``autowrap`` (defaul option) the symbolic expression will be converted, upon linking the :py:class:`.Metric` to the :py:class:`GeodesicEngine`, in pre-compiled C low-level callable with a consitent gain in performance of single function-calls. If ``lambdify`` the geodesic equations will not be compiled as C callables and will be converted to Python callables, with a loss in performances. The :py:class:`GeodesicEngine` will fallback to a ``lambdify`` whenever auxiliary py-functions are used to define the :py:class:`.Metric` object (see :doc:`create_metric` for an example).
            :type backend: Literal["autowrap", "lambdify"]
            :param integrator: Specifies the numerical integration schemes used to carry out the geodesic integration. See :doc:`integrators`. Default is ``rkf45`` correspoding to the :py:class:`.RungeKuttaFehlberg45` integrator.
            :type integrator: Literal['rkf45', 'dp45', 'ck45', 'rkf78']
        """
        if metric == None:
            if len(Metric.instances) > 0:
                if isinstance(Metric.instances[-1], Metric):
                    metric = Metric.instances[-1]
                    logging.warning(f"No Metric object passed to the Geodesic Engine constructor. Using last initialized Metric ({metric.name}) instead.")
            else:
                raise ValueError("No Metric found, initialize one and pass it as argument to the GeodesicEngine constructor.")
        
        if not integrator in (valid_integrators := get_args(AVAILABLE_INTEGRATORS)):
            raise ValueError(f"'integrator' must be one of {valid_integrators}")
        
        self.link_metric(metric, backend)
        self._integrator = integrator
        GeodesicEngine.instances.append(self)
    
    def link_metric(self, g: Metric, backend: _BACKEND_TYPES):
        
        if not backend in (valid_backends := get_args(_BACKEND_TYPES)):
            raise ValueError(f"'backend' must be one of {valid_backends}")
        
        logging.info(f"Linking {g.name} to the Geodesic Engine")

        self.metric = g
        self.eq_x = g.eq_x
        self.eq_u = g.eq_u
        
        self.metric._geodesic_engine_linked = True
        self.metric.geodesic_engine = self

        if (backend == "lambdify") or len(self.metric.get_parameters_functions()) > 0:
            self._wrapper = "lambdify"
        else:
            self._wrapper = "autowrap"

        self._motion_eq_f = []

        for eq in [*self.eq_x, *self.eq_u]:
            if self._wrapper == "autowrap":
                try:
                    self._motion_eq_f.append(autowrap(self.metric.subs_functions(eq), backend='cython', args = [*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()]))
                except OSError:
                    logging.warning("Falling back to 'lambdify' backend because the current platform is not compatible with 'autowrap'. This may affect performances.")
                    self._wrapper = "lambdify"
                    self._motion_eq_f.append(lambdify([*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()], self.metric.subs_functions(eq)))
            elif self._wrapper == "lambdify":
                self._motion_eq_f.append(lambdify([*self.metric.x, *self.metric.u, *self.metric.get_parameters_symb()], self.metric.subs_functions(eq)))

        def _f_eq(tau, xu):
            return np.array([self._motion_eq_f[i](*xu, *self.metric.get_parameters_val()) for i in range(8)])

        self.motion_eq = _f_eq
        
        u0_f_timelike = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_timelike))
        self.u0_f_timelike = lambda initial_x, u1, u2, u3: abs(u0_f_timelike(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))
        u1_f_timelike = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u1_s_timelike))
        self.u1_f_timelike = lambda initial_x, u0, u2, u3: abs(u1_f_timelike(*initial_x, u0, u2, u3, *self.metric.get_parameters_val()))
        u2_f_timelike = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[1], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u2_s_timelike))
        self.u2_f_timelike = lambda initial_x, u0, u1, u3: abs(u2_f_timelike(*initial_x, u0, u1, u3, *self.metric.get_parameters_val()))
        u3_f_timelike = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[1], self.metric.u[2], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u3_s_timelike))
        self.u3_f_timelike = lambda initial_x, u0, u1, u2: abs(u3_f_timelike(*initial_x, u0, u1, u2, *self.metric.get_parameters_val()))
        
        u0_f_null = lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u0_s_null))
        self.u0_f_null = lambda initial_x, u1, u2, u3: abs(u0_f_null(*initial_x, u1, u2, u3, *self.metric.get_parameters_val()))
        u1_f_null = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u1_s_null))
        self.u1_f_null = lambda initial_x, u0, u2, u3: abs(u1_f_null(*initial_x, u0, u2, u3, *self.metric.get_parameters_val()))
        u2_f_null = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[1], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u2_s_null))
        self.u2_f_null = lambda initial_x, u0, u1, u3: abs(u2_f_null(*initial_x, u0, u1, u3, *self.metric.get_parameters_val()))
        u3_f_null = lambdify([*self.metric.x, self.metric.u[0], self.metric.u[1], self.metric.u[2], *self.metric.get_parameters_symb()], self.metric.subs_functions(g.u3_s_null))
        self.u3_f_null = lambda initial_x, u0, u1, u2: abs(u3_f_null(*initial_x, u0, u1, u2, *self.metric.get_parameters_val()))
        
        self.stopping_criterion = StoppingCriterionList([])
        
        logging.info("Metric linking complete.")

    def set_stopping_criterion(self, expr: str, exit_str: str = "none"):
        r"""
            Creates a :py:class:`.StoppingCriterion` given the symbolic expression contained in the ``expr`` argument and sets is as the sole stopping criterion for the :py:class:`GeodesicEngine`.
            
            :param expr: a string to be sympy-parsed containing the expression that will be evaluated at each integration time-step.
            :type expr: str
            :param exit_str: the exit message that will be attached to the :py:class:`.Geodesic` when the stopping condition is met (*i.e.* when the condition in ``expr`` return ``False``).
            :type exit_str: str
        """
        self.stopping_criterion = StoppingCriterionList([])
        self.add_stopping_criterion(expr, exit_str)
        
    def add_stopping_criterion(self, expr: str, exit_str: str = "none"):
        r"""
            Adds a :py:class:`.StoppingCriterion` to the :py:class:`.StoppingCriterionList` of the :py:class:`.GeodesicEngine`. Has the same parameters as :py:func:`set_stopping_criterion`
        """
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
        
    def integrate(self, geo: 'Geodesic', tauf: float, initial_step: float, verbose: bool = False, direction: Literal["fw", "bw"] = "fw", **integrator_kwargs) -> None:
        """
            The main function for the numerical integration. Accepts an initialized :py:class:`.Geodesic` (*i.e.* with appropriately set initial space-time position and 4-velocity), performs the numerical integration of the geodesic equations up to the provided final value of the affine parameter ``tauf`` or until a stopping condition is met, and returns to the input :py:class:`.Geodesic` object the integrated numerical array.
            
            See :doc:`integrate_geodesic` for a complete tutorials.
            
            :param geo: the :py:class:`.Geodesic` to be integrated.
            :type geo: Geodesic
            :param tauf: the value of the final proper time at which to stop the numerical integration.
            :type tauf: float
            :param initial_step: the value of the initial proper time step.
            :type initial_step: float
            :param verbose: whether to log the integration progress to the standard output or not.
            :type verbose: bool
            :param direction: if "fw" the integration is carried on forward in proper time; if "bw" the integration is carried on backward in proper time. The correct signs for ``tauf`` and ``initial_step`` will be assigned automatically.
            :type direction: Literal["fw", "bw"]
            :param integrator_kwargs: the kwargs to be passed to the integrator.
        """
        if not direction in ["fw", "bw"]:
            raise TypeError("direction must be either 'fw' or 'bw'.")
        if verbose:
            logging.info("Starting integration.")
            
        integrator = integrators.get_integrator(self._integrator, self.motion_eq, stopping_criterion = self.stopping_criterion, **integrator_kwargs)

        if direction == "bw":
            h = -abs(initial_step)
            tauf = -abs(tauf)
        else:
            h = abs(initial_step)
            tauf = abs(tauf)

        if verbose:
            time_start = time.perf_counter()

        tau, xu, exit = integrator.integrate(0, tauf, np.array([*geo.initial_x, *geo.initial_u]), h)
                
        if verbose:
            time_elapsed = (time.perf_counter() - time_start)
            logging.info(f"Integration completed in {time_elapsed:.5} s with result '{exit}'.")

        geo.tau = tau
        geo.x = np.stack(xu[:,:4])
        geo.u = np.stack(xu[:,4:])
        geo.exit = exit

class StoppingCriterion:
    r"""
        An helper class to deal with stopping criterion. Requires as input the lambdified callable (``stopping_criterion``) built from a sympy symboic expression (``expr``) and can be called on a given geodesic step to return either ``True`` if the condition in the expression is verified or ``False`` when it is not, stopping the integration. It also requires an exit message (``exit_str``), useful to flag a geodesic that fires the stopping criterion (*i.e.* a geodesic that ends in an horizon).
    """
    def __init__(self, stopping_criterion: Callable[[npt.ArrayLike], bool], expr: str, exit_str: str):
        self.stopping_criterion = stopping_criterion
        self.exit = exit_str
        self.expr = expr
        
    def __repr__(self):
        return f"<StoppingCriterion: {self.expr}>"
    
    def __call__(self, *geo: npt.ArrayLike) -> bool:
        return self.stopping_criterion(*geo)

class StoppingCriterionList:
    r"""
        An aggregator of multiple :py:class:`StoppingCriterion` objects. When called on a geodesic it tests multiple stopping criterions on the last step and returns ``False`` if at least one of the ``stopping_criterions`` is falsy. In that case stores the ``exit`` of the stopping criterion that fired the condition.
    """
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
                break

        return check