Define your own-spacetime
================================================================

In this tutorial we will go through all the different methods in PyGRO for the generation of a :meth:`~pygro.metric_engine.Metric` object that describes a particular space-time metric.
For the sake of simplicity, we will devote this tutorial to the Schwarzschild solution, describing the spherically symmetric spacetime around a point-like particle with mass :math:`M`.
When expressed in Schwarzschild coordinates :math:`(t, r, \theta, \phi)` the line element describing the geometry of this spacetime is described by

.. math::
    ds^2 = -\left(1-\frac{2M}{r}\right)dt^2+\left(1-\frac{2M}{r}\right)^{-1}dr^2+r^2(d\theta^2+\sin^2\theta d\phi^2),

where we have assumed that :math:`G=c=1` and, hence, the radial coordinate and spatial distances are expressed in units of gravitational radii :math:`r_g = GM/c^2`.

Purely symbolic approach
------------------------
In this approach, the :meth:`~pygro.metric_engine.Metric` object will be generated starting from a line element (``str``) which is a function that depends **explicitly** only on the space-time coordinates
and on a given number of ``constant`` parameters. This means that no *auxiliary* function of the space-time coordinates is introduced, as will be done in the :ref:`Auxiliary expressions approach <aux-expr>`
or :ref:`Auxiliary functions approach <aux-func>`.
In order to initialize the :meth:`~pygro.metric_engine.Metric`, we use the *functional approach* for its initialization (see the documentation for the :meth:`~pygro.metric_engine.Metric` object).
We define a ``list`` of ``str`` for the spacetime coordinates, and we express the ``line_element`` as a function of such coordinates :math:`\{x^\mu\}` and of their infinitesimal increment :math:`\{dx^\mu\}`, indicated with a ``d`` as prefix
(e.g. for coordinate ``theta`` the increment is ``dtheta``). Additionally, the ``transform_functions`` list is defined, which contains the symbolic expressions for transformation functions
from the spacetime coordinates in which the ``line_element`` is expressed to pseudo-cartesian coordinates :math:`(t, x, y, z)` that are useful to :doc:`visualize`.

.. tip::
    Since the ``line_element`` is converted into a ``sympy`` expression, a good way to check whether it has been correctly typed,
    is to apply the ``pygro.parse_expr`` function on the ``line_element`` and check that the mathematical expression is properly interpreted.

.. code-block::
    
    import pygro

    name = "Schwarzschild spacetime"
    coordinates = ["t", "r", "theta", "phi"]

    transform_functions = [
        "t",
        "r*sin(theta)*cos(phi)",
        "r*sin(theta)*sin(phi)",
        "r*cos(theta)"
    ]

    line_element = "-(1-2*M/r)*dt**2+1/(1-2*M/r)*dr**2+r**2*(dtheta**2+sin(theta)**2*dphi**2)"

    metric = pygro.Metric(
        name = name,
        coordinates = coordinates,
        line_element = line_element,
        transform = transform_functions
    )

Note that we have passed an additional argument to the Metric constructor ``(..., M = 1)`` by which we have set to unity the value of the parameter :math:`M` in the metric.
In PyGRO constant parameters should always be assigned a numerical value. If no argument ``M`` is passed to the constructor, the user will be prompted to insert one as input:

.. code-block:: none

    >>> Insert value for M: 

During initialization the code will inform the user about the current state of initialization through the standard output:

.. code-block:: none

    Calculating inverse metric...
    Calculating symbolic equations of motion:
    - 1/4
    - 2/4
    - 3/4
    - 4/4
    Adding to class a method to get initial u_0...
    The metric_engine has been initialized.

The :meth:`~pygro.metric_engine.Metric` performs tensorial operations on the newly generated metric tensor :math:`g_{\mu\nu}` (accessible via :attr:`Metric.g`) for computing:

* The inverse metric, accessible via :attr:`Metric.g_inv`;
* The geodesic equations, representing the right-hand side in equation
    .. math::

        \ddot{x}^\mu = \Gamma^{\mu}_{\nu\rho}\dot{x}^\nu\dot{x}^\rho
        
  where, :math:`\Gamma^{\mu}_{\nu\rho}` are the Christoffel symbols accessible via :meth:`~pygro.metric_engine.Metric.Christoffel`.
  These four equations are stored into a list accessible via :attr:`Metric.eq_u`.
* Two symbolic algebraic expressions for the :math:`\dot{x}^0` component of the four velocity derived from the normalization conditions:
    .. math::

        g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = \left\{\begin{array}{ll}
        &-1&\qquad\textrm{time-like curve}\\
        &0&\qquad\textrm{null curve}\\
        \end{array}\right.

  These are particularly useful when one needs to retrieve the time-like component of the four-velocity of a massive particle (or, equivalently, the time-like component of a photon wave-vector)
  knowing the spatial components of the velocity (which is usually the case). See :doc:`integrate_geodesic` for a working example.


Auxiliary expressions approach
-------------------------------
.. _aux-expr:

askfdmaslkfnaslkjfnasknflkas

Auxiliary functions approach
-------------------------------
.. _aux-func:


asòfjaskfjaslkfjlk