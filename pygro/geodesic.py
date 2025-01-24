from typing import Literal, Optional, Tuple, get_args, Sequence, Union
import numpy as np
from pygro import GeodesicEngine

import logging

_GEODESIC_TYPE = Literal['time-like', 'null']
_VALID_GEODESIC_TYPE: Tuple[_GEODESIC_TYPE, ...] = get_args(_GEODESIC_TYPE)

class Geodesic():
    r"""
        The ``pygro`` representation of a geodesic. It is used to define the geodesic type (either time-like or null), initial data and stores the results of the numerical integration, perfromed by the :py:class:`.GeodesicEngine`. 
        
        See :doc:`integrate_geodesic` for more details on the actual integration procedure.
        
        :ivar initial_x: The starting position of the geodesic. It is a 4-dimensional `numpy.ndarray` that can be set using the helper function :py:func:`set_starting_position`.
        :vartype initial_x: np.ndarray
        :ivar initial_u: The starting 4-velocity of the geodesic. It is a 4-dimensional `numpy.ndarray` that can be set using the helper function :py:func:`set_starting_4velocity` which enforces the appropriate normalization condition depending on the geodesic type.
        :vartype initial_u: np.ndarray
    """
    def __init__(self, type: _GEODESIC_TYPE, engine: Optional[GeodesicEngine] = None, verbose: bool = True):
        r"""
            The :py:class:`Geodesic` constructor accepts the following arguments:
            
            :param type: The geodesic normalization. It accepts strings with either ``"time-like"`` (:math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = -1`) or ``"null"`` (:math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0`). 
            :type type: Literal['time-like', 'null']
            :param engine: The :py:class:`.GeodesicEngine` object to link to the :py:class:`Geodesic`. If not provided, the :py:class:`Geodesic` will be linked to the last initialized :py:class:`.GeodesicEngine`.
            :type engine: Optional[GeodesicEngine]
            :param verbose: Specifies whether to log information on the geodesic status to the standard output.
            :type verbose: bool
        """
        if type not in _VALID_GEODESIC_TYPE:
            raise TypeError("Geodesic must either be 'null' or 'time-like'.")
        
        if not engine:
            engine = GeodesicEngine.instances[-1]
            logging.warning(f"No GeodesicEngine object passed to the Geodesic constructor. Using last initialized GeodesicEngine instead")
            
        if not isinstance(engine, GeodesicEngine):
            raise TypeError("No GeodesicEngine found, initialize one and pass it as argument to the Geodesic constructor")
        
        self.engine = engine
        self.metric = engine.metric
        
        self._type = type
        self._verbose = verbose
        
        self._initial_x = None
        self._initial_u = None
        
        self._tau = np.empty(0)
        self._x = np.empty((0,4))
        self._u = np.empty((0,4))
    
    @property
    def type(self) -> _GEODESIC_TYPE:
        return self._type
        
    @property
    def initial_x(self) -> np.ndarray:
        return self._initial_x

    @property
    def initial_u(self) -> np.ndarray:
        return self._initial_u
    
    @property
    def x(self) -> np.ndarray:
        return self._x
    
    @property
    def u(self) -> np.ndarray:
        return self._u
    
    @property
    def tau(self) -> np.ndarray:
        return self._tau
    
    @initial_x.setter
    def initial_x(self, initial_x: Sequence[Union[int, float]]):
        if not isinstance(initial_x, (list, tuple, np.ndarray)):
            raise TypeError("initial_x must be a sequence (list or tuple).")
        if len(initial_x) != 4:
            raise ValueError("initial_x must contain exactly four elements.")
        if not all(isinstance(num, (int, float)) for num in initial_x):
            raise TypeError("All elements in initial_x must be numbers (int or float).")
        
        self._initial_x = np.array(initial_x)
        self._tau = np.empty(0)
        self._x = np.empty((0,4))
        self._u = np.empty((0,4))
    
    @initial_u.setter
    def initial_u(self, initial_u: Sequence[Union[int, float]]):
        if not isinstance(initial_u, (list, tuple, np.ndarray)):
            raise TypeError("initial_u must be a sequence (list or tuple).")
        if len(initial_u) != 4:
            raise ValueError("initial_u must contain exactly four elements.")
        if not all(isinstance(num, (int, float)) for num in initial_u):
            raise TypeError("All elements in initial_u must be numbers (int or float).")
        
        if self.initial_x is None:
            raise ValueError("You must use set_starting_point before initializing the 4-velocity.")
        
        self._tau = np.empty(0)
        self._x = np.empty((0,4))
        self._u = np.empty((0,4))
        self._initial_u = np.array(initial_u)
        
    @x.setter
    def x(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("x must be a numpy array.")
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError("x array must have shape (N, 4), where N can be any number.")
        self._x = value
    
    @u.setter
    def u(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("u must be a numpy array.")
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError("u array must have shape (N, 4), where N can be any number.")
        self._u = value
        
    @tau.setter
    def tau(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("u must be a numpy array.")
        if value.ndim != 1:
            raise ValueError("u array must be a one-dimensional array")
        self._tau = value
        
    def set_starting_point(self, x0: Union[float, int], x1: Union[float, int], x2: Union[float, int], x3: Union[float, int]):
        r"""
            Sets the initial values of the space-time coordinates from which to start the integration.
        """
        if self._verbose:
            logging.info("Setting starting point")
        self.initial_x = np.array([x0, x1, x2, x3])
    
    def get_initial_u0(self, u1, u2, u3):
        if self._type == "null":
            u_sol = self.engine.u0_f_null(self.initial_x, u1, u2, u3)
        elif self._type == "time-like":
            u_sol = self.engine.u0_f_timelike(self.initial_x, u1, u2, u3)
            
        if np.isnan(u_sol):
            raise ValueError("The 4-velocity cannot be normalized, returned NaN")
        
        return u_sol
    
    def get_initial_u1(self, u0, u2, u3):
        if self._type == "null":
            u_sol = self.engine.u1_f_null(self.initial_x, u0, u2, u3)
        elif self._type == "time-like":
            u_sol = self.engine.u1_f_timelike(self.initial_x, u0, u2, u3)
            
        if np.isnan(u_sol):
            raise ValueError("The 4-velocity cannot be normalized, returned NaN")
        
        return u_sol
    
    def get_initial_u2(self, u0, u1, u3):
        if self._type == "null":
            u_sol = self.engine.u2_f_null(self.initial_x, u0, u1, u3)
        elif self._type == "time-like":
            u_sol = self.engine.u2_f_timelike(self.initial_x, u0, u1, u3)
        
        if np.isnan(u_sol):
            raise ValueError("The 4-velocity cannot be normalized, returned NaN")
        
        return u_sol
    
    def get_initial_u3(self, u0, u1, u2):
        if self._type == "null":
            u_sol = self.engine.u3_f_null(self.initial_x, u0, u1, u2)
        elif self._type == "time-like":
            u_sol = self.engine.u3_f_timelike(self.initial_x, u0, u1, u2)
        
        if np.isnan(u_sol):
            raise ValueError("The 4-velocity cannot be normalized, returned NaN")
        
        return u_sol
    
    def set_starting_4velocity(self, u0: Optional[float] = None, u1: Optional[float] = None, u2: Optional[float] = None, u3: Optional[float] = None):
        r"""
            Sets the initial values of the components of the 4-velocity, enforcing the normalization conditions. For this reason, only three out of the four components ``u[i]`` of the 4-velocity must be specified, and the remaining one will be automatically computed to satistfy the normalization condition.
            
            To override this behaviour, you can directly set the ``.initial_u`` property of the :py:class:`Geodesic`.
        """
        check = sum(u is not None for u in [u0, u1, u2, u3])
        
        if not check == 3:
            raise ValueError("Specify three components of the initial 4-velocity.")
        
        if u0 is None:
            u0 = self.get_initial_u0(u1, u2, u3)
        if u1 is None:
            u1 = self.get_initial_u1(u0, u2, u3)
        if u2 is None:
            u2 = self.get_initial_u2(u0, u1, u3)
        if u3 is None:
            u3 = self.get_initial_u3(u0, u1, u2)
        if self._verbose:
            logging.info("Setting initial 4-velocity.")
        self.initial_u = np.array([u0, u1, u2, u3])