Interpolators
=============

PyGRO implements several interpolation strategies for the integrated geodesic. The base interpolator class offers a way to compute the integrate geodesic at each value of the affine parameter on the integration interval. Whenever possible, :py:class:`~pygro.integrators.Integrator` classes implement dense output tailored to the order of the numerical integration involved in the scheme.

When the dense output for the specific class is not available, it will fall back to cubic interpolation using Hermite polynomials which guarantees third-order accuracy for integrators of order :math:`p\geq3`. 

.. automodule:: pygro.interpolators
    :members: