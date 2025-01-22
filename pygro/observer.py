import sympy as sp
import numpy as np

from typing import Sequence, Union, Optional
from pygro.metric_engine import Metric
from pygro.geodesic import _GEODESIC_TYPE, _VALID_GEODESIC_TYPE

import logging

class Observer:
    '''
        The :py:class:`Observer` class represent PyGRO representation of physical observers in space-time. It is built by specifying either a frame or a co-frame uniquely identifying a tetrad in space-time. This can than be used to fire geodesics from the observer's position giving a physical meaning to the intial values of the 4-velocity components for the integrated geodesic.
    '''
    def __init__(self, metric: Optional[Metric], x: Sequence[Union[int, float]], frame: Optional[list[str]] = None, coframe: Optional[list[str]] = None):
        r'''
            Initializes the :py:class:`Observer` class. Accepts the following arguments:
            
            :param metric: The :py:class:`.Metric` object to link to the :py:class:`GeodesicEngine` and from which the geodesic equations and all the related symbolic quantities are retrieved. If not provided, the :py:class:`GeodesicEngine` will be linked to the last initialized :py:class:`.Metric`.
            :type metric: Metric
            :param x: The space-time position of the observer (must be a 4-dimensional array of numbers).
            :type x: Sequence[Union[int, float]]
            :param frame/coframe: The symbolic definitions of either the observer's frame or coframe in the tetrad formalism (accepts a list of four strings, one for each tetrad). See the tutorial :doc:`define_observer` for more details.
            :type frame/coframe: list[str]
            
        '''
        if not isinstance(x, (list, tuple, np.ndarray)):
            raise TypeError("'x''must be a sequence (list, tuple or numpy.ndarray).")
        if len(x) != 4:
            raise ValueError("'x' must contain exactly four elements.")
        
        if metric == None:
            if len(Metric.instances) > 0:
                if isinstance(Metric.instances[-1], Metric):
                    metric = Metric.instances[-1]
                    logging.warning(f"No Metric object passed to the Geodesic Engine constructor. Using last initialized Metric ({metric.name}) instead.")
            else:
                raise ValueError("No Metric found, initialize one and pass it as argument to the GeodesicEngine constructor.")
        
        self.metric = metric
        self.x = np.array(x)
                
        check = sum(f is not None for f in [frame, coframe])
        
        if not check == 1:
            raise ValueError('You must specify one (and only one) between frame and coframe for the observer.')
                
        if frame is not None:
            self._get_frame(frame)
        if coframe is not None:
            self._get_coframe(coframe)
    
    def _get_coframe(self, coframe: list[str]):
        
        if len(coframe) != 4:
                raise ValueError('coframe should be a 4-dimensional list of strings')
        
        coframe_symb = []
        self.coframe_matrix = sp.zeros(4, 4)
    
        try:
            for coframe_i in coframe:
                coframe_symb.append(sp.expand(sp.parse_expr(coframe_i)))
        except:
            raise ValueError("Please insert a valid expression for the coframes components.")
        else:
            for i, coframe_i in enumerate(coframe_symb):
                for j, dx in enumerate(self.metric.dx):
                        self.coframe_matrix[i, j] = coframe_i.coeff(dx,1)
            
            self.frame_matrix = self.coframe_matrix.inv()
    
    def _get_frame(self, frame : list[str]):
        if len(frame) != 4:
                raise ValueError('frame should be a 4-dimensional list of strings')
        
        frame_symb = []
        self.frame_matrix = sp.zeros(4, 4)
    
        try:
            for frame_i in frame:
                frame_symb.append(sp.expand(sp.parse_expr(frame_i)))
        except:
            raise ValueError("Please insert a valid expression for the frames components.")
        else:
            for i, frame_i in enumerate(frame_symb):
                for j, x in enumerate(self.metric.x):
                        self.frame_matrix[i, j] = frame_i.coeff(sp.symbols("e"+str(x)),1)
            
            self.coframe_matrix = self.frame_matrix.inv()
            
    def convert_3vector(self, vector: Sequence[Union[int, float]], type: _GEODESIC_TYPE):
        r'''
            Converts a 3-vector in the reference frame of the :py:class:`Observer` into a 4-vector in space-time. The user must specify the components of the 3-vector ``v`` and the desired 4-vector ``type`` that can either be ``time-like`` or ``geodesic``.
            
            :param vector: A 3-dimensional sequence of numbers corresponding to the components of the 3-vector in the Observer's reference frame. 
            :type vector: Sequence[Union[int, float]]
            :param type: The vector normalization. It accepts strings with either ``"time-like"`` (:math:`g_{\mu\nu}u^\mu u^\nu = -1`) or ``"null"`` (:math:`g_{\mu\nu}u^\mu u^\nu = 0`). 
            :type type: Literal['time-like', 'null']
            
        '''
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise TypeError("'x''must be a sequence (list, tuple or numpy.ndarray).")
        if len(vector) != 3:
            raise ValueError("'vector' must be a 3-vector, containing exactly three elements.")
        
        if type not in _VALID_GEODESIC_TYPE:
            raise TypeError("Vector type must either be 'null' or 'time-like'.")
        
        u_123 = sp.lambdify([*self.metric.x, *self.metric.get_parameters_symb()], self.metric.subs_functions(self.frame_matrix[1:, 1:]))(*self.x, *self.metric.get_parameters_val())@np.array(vector)
        
        if type == "time-like":
            u0 = abs(sp.lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(self.metric.u0_s_timelike))(*self.x, *u_123, *self.metric.get_parameters_val()))
        elif type == "null":
            u0 = abs(sp.lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(self.metric.u0_s_null))(*self.x, *u_123, *self.metric.get_parameters_val()))
        
        return np.append(u0, u_123)
    
    def _from_frame_vector(self, theta_obs: float, phi_obs: float, type = _GEODESIC_TYPE, v: Optional[float] = None) -> Sequence[float]:
        
        if not isinstance(theta_obs, (float, int)) or not isinstance(phi_obs, (float, int)):
            raise TypeError("Angles theta and phi must be numbers.")
        
        if type == "time-like":
            if v is None:
                raise ValueError("You must set the modulus 'v' of the 3-velocity in the Observer reference frame for a time-like vector")
            if v < 0:
                raise ValueError("3-velocity must be positive")
        else:
            v = 1

        x_obs: float = v*np.cos(theta_obs)*np.cos(phi_obs)
        y_obs: float = v*np.cos(theta_obs)*np.sin(phi_obs)
        z_obs: float = v*np.sin(theta_obs)
        
        return x_obs, y_obs, z_obs
    
    def from_f1(self, theta_obs: float, phi_obs: float, type = _GEODESIC_TYPE, v: Optional[float] = None):
        r'''
            Returns a 4-vector corresponding to the initial 4-velocity for a time-like or null geodesic (depending on the ``type`` argument) fired with angles ``theta_obs`` and ``phi_obs`` from the :math:`f_1` vector. See :doc:`define_observer` for an illustrative example of this functionality.
            
            :param theta_obs: The angle :math:`\theta_{\rm obs}` in the local observer's frame.
            :type theta_obs: float
            :param phi_obs: The angle :math:`\phi_{\rm obs}` in the local observer's frame.
            :type phi_obs: float
            :param type: The vector normalization. It accepts strings with either ``"time-like"`` (:math:`g_{\mu\nu}u^\mu u^\nu = -1`) or ``"null"`` (:math:`g_{\mu\nu}u^\mu u^\nu = 0`). 
            :type type: Literal['time-like', 'null']
            :param v: The modulus of the spatial velocity of the geodesic. To be specified in the ``time-like case.``
            :type v: float
        '''
        x, y, z = self._from_frame_vector(theta_obs, phi_obs, type, v)
        return self.convert_3vector([x, y, z], type)
    
    def from_f2(self,  theta_obs: float, phi_obs: float, type = _GEODESIC_TYPE, v: Optional[float] = None):
        r'''
            As :py:meth:`~.pygro.observer.Observer.from_f1` method, but for the vector :math:`f_2`.
        '''
        x, y, z = self._from_frame_vector(theta_obs, phi_obs, type, v)
        return self.convert_3vector([z, x, y], type)
    
    def from_f3(self,  theta_obs: float, phi_obs: float, type = _GEODESIC_TYPE, v: Optional[float] = None):
        r'''
            As :py:meth:`~.pygro.observer.Observer.from_f1` method, but for the vector :math:`f_3`.
        '''
        x, y, z = self._from_frame_vector(theta_obs, phi_obs, type, v)
        return self.convert_3vector([z, y, x], type)