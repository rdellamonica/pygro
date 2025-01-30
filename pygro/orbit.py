from pygro import Geodesic, GeodesicEngine, Metric
from scipy.optimize import fsolve, root
import sympy as sp
import numpy as np
import sympy as sp
import logging
from .utils.rotations import *

from typing import Optional, Literal

class Orbit:
    """
        Base class for integrating physical orbits in spherically symmetric spacetimes. It is a wrapper around the :py:class:`.Geodesic` object for the ``time-like`` case with additional methods to assign initial conditions based on the standard Keplerian parameters in spherically symmetric space-times.
    """
    def __init__(self, geo_engine: Optional[GeodesicEngine] = None, verbose: bool = False, orbital_parameters : Optional[dict] = None, radial_coordinate : Optional[sp.Basic] = None, latitude : Optional[sp.Basic] = None, longitude : Optional[sp.Basic] = None):
        r"""
            The :py:class:`.Orbit` constructor accepts the following arguments:
            
            :param geo_engine: The :py:class:`.GeodesicEngine` object to link to the :py:class:`.Orbit`. If not provided, the :py:class:`Geodesic` will be linked to the last initialized :py:class:`.GeodesicEngine`.
            :type geo_engine: Optional[GeodesicEngine]
            :param verbose: Specifies whether to log information on the geodesic status to the standard output.
            :type verbose: bool
            :param orbital_parameters: An optional dictionary to be passed to the :py:meth:`~pygro.orbit.Orbit.set_orbital_parameters` method. See the documentation of this method for the exact details of what to include in the dictionary.
            :type orbital_parameters: Optional[dict]
            :param radial_coordinate: When initializing the :py:class:`.Orbit`, it will assume by the default that the radial coordinate in the metric will be the first component of the coordinate array (*i.e.* the content of ``Metric.x[1]``). This can be overridden by specifically passing the symbol corresponding to the radial coordinate.
            :type radial_coordinate: Optional[sp.Basic]
            :param latitude: When initializing the :py:class:`.Orbit`, it will assume by the default that the latitude angular coordinate (:math:`\theta` in the usual Schwarzschild coordinates, see :doc:`create_metric`) in the metric will be the second component of the coordinate array (*i.e.* the content of ``Metric.x[2]``). This can be overridden by specifically passing the symbol corresponding to the latitude.
            :type latitude: Optional[sp.Basic]
            :param longitude: When initializing the :py:class:`.Orbit`, it will assume by the default that the longitude angular coordinate (:math:`\phi` in the usual Schwarzschild coordinates, see :doc:`create_metric`) in the metric will be the third component of the coordinate array (*i.e.* the content of ``Metric.x[3]``). This can be overridden by specifically passing the symbol corresponding to the longitude.
            :type longitude: Optional[sp.Basic]
            
            After initialization, the :py:class:`.Orbit` instance has the following attributes:

            :ivar params: The dictionary containing the orbital parameters
            :vartype params: dict
            :ivar V_eff: A callable which returns the effective potential of the orbit as a function of the orbital energy, angular momentum and of the radial coordinate :math:`V_\textrm{eff} = V_\textrm{eff}(E, L, r)`.
            :vartype V_eff: Callable
            :ivar E: The value of the orbital energy of the orbit that corresponds to the given orbital parameters.
            :vartype E: float
            :ivar L: The value of the orbital angular momentum of the orbit that corresponds to the given orbital parameters.
            :vartype L: float
        """
        if not geo_engine:
            geo_engine = GeodesicEngine.instances[-1]
            logging.warning(f"No GeodesicEngine object passed to the Geodesic constructor. Using last initialized GeodesicEngine instead")
            
        if not isinstance(geo_engine, GeodesicEngine):
            raise TypeError("No GeodesicEngine found, initialize one and pass it as argument to the Orbit constructor")

        self.geo_engine : GeodesicEngine = geo_engine
        self.metric : Metric = geo_engine.metric
        
        self._t, self._r, self._theta, self._phi = self.metric.x
        self._u_t, self._u_r, self._u_theta, self._u_phi = self.metric.u
        
        if self.metric.Lagrangian().diff(self._t) != 0:
            raise ValueError("Space-time is not stationary.")
        
        if self.metric.Lagrangian().diff(self._phi) != 0:
            raise ValueError("Space-time is not azimuthally symmetric.")
        
        if (radial_coordinate is not None):
            if (radial_coordinate in self.metric.x):
                self._r = radial_coordinate
                self._u_r = self.metric.u[self.metric.x.index(radial_coordinate)]
            else:
                raise ValueError(f"Cannot set {str(radial_coordinate)} as radial coordinate")
        
        if (latitude is not None):
            if (latitude in self.metric.x):
                self._theta = latitude
                self._u_theta = self.metric.u[self.metric.x.index(latitude)]
            else:
                raise ValueError(f"Cannot set {str(latitude)} as latitude")
        
        if (longitude is not None):
            if (longitude in self.metric.x):
                self._phi = longitude
                self._u_phi = self.metric.u[self.metric.x.index(longitude)]
            else:
                raise ValueError(f"Cannot set {str(longitude)} as longitude")
        
        if orbital_parameters:
            self.set_orbital_parameters(**orbital_parameters)
            
        self.geo = Geodesic('time-like', self.geo_engine, verbose)
        
    @property
    def params(self) -> dict:
        return self._params
    
    @params.setter
    def params(self, params):
        self._params : dict = params
        for k, v in params.items():
            setattr(self, k, v)
    
    def set_orbital_parameters(self, t_P: Optional[float] = None, a: Optional[float] = None, e: Optional[float] = None, i: Optional[float] = None, omega: Optional[float] = None, Omega: Optional[float] = None, t_A: Optional[float] = None):
        r"""
            Sets the orbital parameters that uniquely identify the orbit on the equatorial plane. 
            
            For a detailed explaination on how a choice of orbital parameters is translated into intial conditions for the geodesic, see the :doc:`integrating_orbits` tutorial and the example notebooks :doc:`examples/Schwarzschild-ISCO` and :doc:`examples/Schwarzschild-Precession`.
            
            The parametrization that we choose is the usual one in celestial mechanics:
            
            * :math:`t_p` (``t_P``): coordinate time of pericenter passage. Alternatively, one can specify the coordinate time of apocenter passge :math:`t_a` (``t_A``). Assigning one (and only one) of these parameters is **mandatory** and will raise an error if not assigned. Depending on the choice between ``t_P`` and ``t_A`` the integration will be started at pericenter or apocenter.
            * :math:`a` (``a``): The semi-major axis of the orbit in the same units in which the radial coordiante in the metrix is expressed. This parameter is **mandatory** and will raise an error if not assigned.
            * :math:`e` (``e``): The orbital eccentricity, which fixes the radial turning points :math:`r_p=a(1-e)` and :math:`r_a=a(1+e)` (see :doc:`integrating_orbits`). This parameter is **mandatory** and will raise an error if not assigned.
            * :math:`(i,\,\omega,\,\Omega)` (``i, omega, Omega``): the three angular orbital parameters (in radians) which will rotate the initial conditions out of the equatorial plane. See :doc:`integrating_orbits` for a proper definition of these parameters. These parameters are optional. If not specified they will be fixed to zero, leaving the orbit on the equatorial plane (:math:`\theta = \pi/2` or the corresponding latitude coordinate).
            
        """
        
        if a is None:
            raise ValueError("Semi-major not specified.")
        elif a <= 0:
            raise ValueError("Semi-major axis must be a > 0")
        
        if sum(t is not None for t in [t_P, t_A]) != 1:
            raise ValueError("Must specify one (and only one) between time of pericenter passage (t_P) and time of apocenter passage (t_A).")
        
        if e is None:
            raise ValueError("Eccentricity not specified.")
        elif e < 0:
            raise ValueError("Eccentricity must be > 0")
        elif e >= 1:
            raise ValueError("Eccentricity must be <= 1")
        
        if Omega is None:
            Omega = 0
            logging.warning("Longitude of ascending node not specified, set to 0.")
            
        if omega is None:
            omega = 0
            logging.warning("Argument of the pericenter not specified, set to 0.")
        
        if i is None:
            i = 0
            logging.warning("Inclination not specified, set to 0.")
        
        if t_P is not None:
            self._at = "peri"
            self.params = dict(
                t_P = t_P, a = a, e = e, i = i, omega = omega, Omega = Omega
            )
        elif t_A is not None:
            self._at = "apo"
            self.params = dict(
                t_A = t_A, a = a, e = e, i = i, omega = omega, Omega = Omega
            )
            
    def _compute_initial_conditions(self, initial_conditions_tol: float = 1e-12):
        
        r_P = self.a*(1-self.e)
        r_A = self.a*(1+self.e)
        
        Lagrangian = self.metric.subs_functions(self.metric.Lagrangian()).subs([(self._theta, sp.pi/2), (self._u_theta, 0)])
        
        E = -Lagrangian.diff(self._u_t)
        L = Lagrangian.diff(self._u_phi)

        E_s, L_s = sp.symbols("E L")

        u_t_E = sp.solve(E-E_s, self._u_t)[0]
        u_phi_L = sp.solve(L-L_s, self._u_phi)[0]

        u_r2_lagr = sp.solve(Lagrangian.subs([(self._u_t, u_t_E), (self._u_phi, u_phi_L)]) + 1/2, self._u_r**2, simplify=False)[0]

        u_r2_func = sp.lambdify([E_s, L_s, self._r, *self.metric.get_parameters_symb()], u_r2_lagr)
        u_r2_prime_func = sp.lambdify([E_s, L_s, self._r, *self.metric.get_parameters_symb()], u_r2_lagr.diff(self._r))

        u_t_func = sp.lambdify([E_s, L_s, self._r, *self.metric.get_parameters_symb()], u_t_E)
        u_phi_func = sp.lambdify([E_s, L_s, self._r, *self.metric.get_parameters_symb()], u_phi_L)
        
        self.V_eff = sp.lambdify([E_s, L_s, self._r], self.metric.evaluate_parameters(self.metric.subs_functions(u_r2_lagr)))
        
        def u_r2(EL):
            E, L = EL
            if self.e > 0:
                return u_r2_func(E, L, r_P, *self.metric.get_parameters_val()), u_r2_func(E, L, r_A, *self.metric.get_parameters_val())    
            elif self.e == 0:
                return u_r2_prime_func(E, L, r_P, *self.metric.get_parameters_val()), u_r2_func(E, L, r_P, *self.metric.get_parameters_val())
            
        # Computing Energy and Angular momentum from the keplerian orbit as initial guessess for the root search
        
        GM = float(sp.limit(self.metric.evaluate_parameters((self.metric.subs_functions(self.metric.g[0,0])+1)/(2/self._r)), self._r, sp.oo))

        keplerian_T = np.sqrt(4*np.pi**2*self.a**3/GM)
        
        keplerian_u_phi_0 = 2*np.pi/keplerian_T*self.a**2/r_P**2*np.sqrt(1-self.e**2)
        keplerian_u_t_0 = self.geo_engine.u0_f_timelike([0, r_P, np.pi/2, 0], 0, 0, keplerian_u_phi_0)

        E_0_proposal = self.metric.get_evaluable_function(E)(r_P, keplerian_u_t_0)
        L_0_proposal = self.metric.get_evaluable_function(L)(r_P, keplerian_u_phi_0)

        sol = root(u_r2, [E_0_proposal, L_0_proposal], tol = initial_conditions_tol, method = 'lm')
        
        self._sol = sol
        
        if (not all(np.array(u_r2(sol.x)) <= initial_conditions_tol)) or self.e*u_r2_prime_func(*sol.x, r_P, *self.metric.get_parameters_val())*u_r2_prime_func(*sol.x, r_A, *self.metric.get_parameters_val()) > 0:
            logging.warning("Could not find stable initial conditions with current orbital parameters. Falling back to approximate Keplerian initial conditions.")
            E_0, L_0 = E_0_proposal, L_0_proposal
        else:
            E_0, L_0 = sol.x
        
        self.E, self.L = E_0, L_0
        
        r_0 = r_P if self._at == "peri" else -r_A
        
        u_t_0 = abs(u_t_func(E_0, L_0, abs(r_0), *self.metric.get_parameters_val()))
        u_phi_0 = abs(u_phi_func(E_0, L_0, abs(r_0), *self.metric.get_parameters_val()))
        
        x0 = r_0
        y0 = 0
        z0 = 0
        
        vx0 = 0
        vy0 = r_0*u_phi_0/u_t_0
        vz0 = 0
        
        r0 = np.array([x0, y0, z0])
        v0 = np.array([vx0, vy0, vz0])
        
        R1 = Rotate_z(self.omega)
        R2 = Rotate_y(self.i)
        R3 = Rotate_z(self.Omega)

        self._r0 = R3@R2@R1@r0
        self._v0 = R3@R2@R1@v0
        self._ut0 = u_t_0
        self._t0 = self.t_P if self._at == "peri" else self.t_A
    
    def integrate(self, tauf : float | int, initial_step : float | int, verbose : bool = False, direction : Literal["bw", "fw"] = "fw", initial_conditions_tol : float = 1e-15, **integration_kwargs):
        """
            A wrapper around the :py:meth:`~pygro.geodesic_engine.GeodesicEngine.integrate` method.
            
            It accepts the same arguments plus:
            
            :param initial_conditions_tol: sets the numerical tolerance for the root finding algorithm of the initial conditions.
            :type initial_conditions_tol: float
            
            This method sets the initial conditions of the integration starting from the orbital parameters. In case the procedure is not successfull it falls back to the classical Keplerian initial conditions (alerting the user that this has been the case).
            
        """
        if hasattr(self, 'params'):
            self._compute_initial_conditions(initial_conditions_tol = initial_conditions_tol)
        else:
            raise Exception("Orbital parameters have not been set.")
        
        r0, theta0, phi0 = cartesian_to_spherical_point(*self._r0)
        vr0, vtheta0, vphi0 = cartesian_to_spherical_vector(*self._r0, *self._v0)
        
        self.geo.set_starting_point(self._t0, r0, theta0, phi0)
        self.geo.initial_u = [self._ut0, vr0*self._ut0, vtheta0*self._ut0, vphi0*self._ut0]
        
        self.geo_engine.integrate(self.geo, tauf, initial_step, verbose, direction, **integration_kwargs)