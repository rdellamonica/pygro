Integrators
===========

Here, we collect the documentations of the ordinary-differential-equations (ODE) integrators that are implemented in PyGRO.

A series of different integration schemes is pre-built in PyGRO. These can be chosen at the moment of defining a :py:class:`.GeodesicEngine`.

In particular, we have implemented a series of adaptive step-size `explicit Runge-Kutta methods <https://en.wikipedia.org/wiki/Runge-Kutta_methods#Explicit_Runge-Kutta_methods>`_:

- **Runge-Kutta-Fehlberg4(5)**: (``integrator = "rkf45"``) embedded method from the Runge-Kutta family of the 4th order with error estiamtion of the 5th order. The implemented version is based on [1]_.
- **Dormand-Prince5(4)**: (``integrator = "dp45"``) embedded method of the 5th order with error estiamtion of the 4th order. The implemented version is based on [2]_. It is the **default** choice in PyGRO when no ``integrator`` argument is passed to the :py:class:`.GeodesicEngine`.
- **Cash-Karp**: (``integrator = "ck45"``) embedded method of the 4th order with error estiamtion of the 5th order. The implemented version is based on [3]_.
- **Runge-Kutta-Fehlberg7(8)**: (``integrator = "rkf78"``) embedded method from the Runge-Kutta family of the 7th order with error estiamtion of the 8th order. The implemented version is based on [4]_.

All the implemented methods refer to the general class of :py:class:`.ExplicitAdaptiveRungeKuttaIntegrator`, whose docuemntation is reported here:

.. autoclass:: pygro.integrators.ExplicitAdaptiveRungeKuttaIntegrator()
    :members:

In the future, we also plan to implement implicit and symplectic integrations schemes.

.. rubric:: References

.. [1] Fehlberg, E (1964). "New high-order Runge-Kutta formulas with step size control for systems of first and second-order differential equations". Zeitschrift für Angewandte Mathematik und  Mechanik. 44 (S1): T17 - T29. `doi:10.1002/zamm.19640441310 <https://doi.org/10.1002%2Fzamm.19640441310>`_.

.. [2] Dormand, J.R.; Prince, P.J. (1980). "A family of embedded Runge-Kutta formulae". Journal of Computational and Applied Mathematics. 6 (1): 19-26. `doi:10.1016/0771-050X(80)90013-3 <https://doi.org/10.1016/0771-050X(80)90013-3>`_

.. [3] J. R. Cash, A. H. Karp. (1990). "A variable order Runge-Kutta method for initial value problems with rapidly varying right hand sides", ACM Transactions on Mathematical Software 16: 201-222. `doi:10.1145/79505.79507 <https://doi.org/10.1145/79505.79507>`_

.. [4] Fehlberg, Erwin (1968) "Classical fifth-, sixth-, seventh-, and eighth-order Runge-Kutta formulas with stepsize control". NASA Technical Report 287. (`PDF <https://ntrs.nasa.gov/api/citations/19680027281/downloads/19680027281.pdf>`_).