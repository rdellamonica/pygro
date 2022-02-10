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
.. _symb:

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

In this section we review a different symbolic approach to generate a :meth:`~pygro.metric_engine.Metric` object which, differently than
before, relies on an auxiliary function which has a closed analytical form. Suppose, for the sake of simplicity, that one desires to generate
the same Schwarzschild metric that has been computed in the :ref:`Purely symbolic approach <symb>`, but instead of defining it purely symbolically,
one wants to write it using the following expression 
 
.. math::
    ds^2 = -A(r)dt^2+\frac{1}{A(r)}dr^2+r^2(d\theta^2+\sin^2\theta d\phi^2),

where:

.. math::
    A(r) = \left(1-\frac{2M}{r}\right).

Clearly the new expression is formally equivalent to that in the previous section and one might think that this reformulation is not useful. However, for much more complicated metrics, having the possibility to inject into the metric auxiliary functions whose actual analytic expression is indicated elsewhere can be really useful and allow for neater formulation of the problem. For this reason, in PyGRO a functionality to accept auxiliary functions as part of the metric expression has been introduced. It can be easily accessed by specifying the auxiliary function and its dependency from the spacetime coordinates (e.g. ``A(r)`` in our case) in the ``line_element`` and later passing as additional keyword argument, whose name is the functional part of the function, to the :meth:`~pygro.metric_engine.Metric` constructor a ``str`` containing the symbolic expression of the function (e.g. ``(..., A = "1-2*M/r")``). Again, any constant parameter that is used in the auxiliary expression must be specified as additional keyword argument (e.g. ``(..., M = 1)``).

Here is what a :meth:`~pygro.metric_engine.Metric` initialization would look like in this case:

.. code-block::

    name = "Schwarzschild spacetime"
    coordinates = ["t", "r", "theta", "phi"]

    transform_functions = [
        "t",
        "r*sin(theta)*cos(phi)",
        "r*sin(theta)*sin(phi)",
        "r*cos(theta)"
    ]

    line_element = "-A(r)*dt**2+1/A(r)*dr**2+r**2*(dtheta**2+sin(theta)**2*dphi**2)"

    A = "1-2*M/r"

    metric = pygro.Metric(
        name = name,
        coordinates = coordinates,
        line_element = line_element,
        transform = transform_functions,
        A = A,
        M = 1
    )

.. note::

    Auxiliary expression can also rely on *other* auxiliary expressions, as long as on metric initialization they are all properly passed to the :meth:`~pygro.metric_engine.Metric` constructor. For example, the previous metric could also be defined as ``line_element = "-A(r)*dt**2+B(r)*dr**2+r**2*(dtheta**2+sin(theta)**2*dphi**2)"``, provided that the initialization is done with ``metric = pygro.Metric(..., line_element = line_element, transform = transform_functions, A = "1-2*M/r", B = "1/A(r)", M = 1)``.

Auxiliary functions approach
-------------------------------
.. _aux-func:

Finally, we have a last approach for the metric initialization, which relies on auxiliary ``pyfunc`` methods as parts of the line element. This approach is particularly useful when we wish to introduce in the metric functions of the coordinates that do not have an analytic expression and rely on, for example, the solution of an integral or on an interpolated/tabulated function which is not available within the ``sympy`` module. This approach allows to use any function defined in the ``__main__`` body of your script as auxiliary function.

.. caution::
    We suggest using the *Auxiliary functions approach* only when strictly dictated by the problem you want to solve, i.e. only if it is necessary to rely on an external function that cannot be expressed symbolically with an analytic expression. This is because PyGRO reaches its best performances when integrating geodesic equations expressed in a completely symbolic way. More specifically, upon linking of a :meth:`~pygro.metric_engine.Metric` element to a :meth:`~pygro.geodesic_engine.GeodesicEngine`, PyGRO makes use of the built-in ``sympy`` method ``autowrap``, which converts the call to a specific symbolic expression into a C-precompiled binary executable. This **drastically** improves the integration performances.

In order to correctly initialize a metric using the *Auxiliary functions approach* the user must take into account the fact that Christoffel symbols and, hence, geodesic equations are computed from the derivatives of the metric coefficients. This means that, while in the purely symbolic approaches the :meth:`~pygro.metric_engine.Metric` deals autonomously with the computation of such derivatives, in the auxiliary functions approach the user should not only pass to the :meth:`~pygro.metric_engine.Metric` constructor the ``pyfunc`` corresponding to the auxiliary functions reported in the line element, but also its derivatives with respect to all the coordinates on which it explicitly depends. These must be passed as keyword arguments to the metric constructor corresponding to the following syntax:

> ``"A(r)" -> Metric(..., A = [...], dAdr = [...])``

It is important to notice that the ``pyfunc`` to pass to the metric must be defined as a method depending on four arguments, one for each coordinate, that has to be ordered exactly as the coordinates of the metric. 

Here, for example, we initialize the same Schwarzschild metric of the previous examples but using the auxiliary functions approach:

.. code-block::

    name = "Schwarzschild spacetime"
    coordinates = ["t", "r", "theta", "phi"]
    line_element = "-A(r)*dt**2+1/A(r)*dr**2+r**2*(dtheta**2+dphi**2)"
    transform = [
        "t",
        "r*sin(theta)*cos(phi)",
        "r*sin(theta)*sin(phi)",
        "r*cos(theta)"
    ]

    def A(t, r, theta, phi):
        M = metric.get_constant("M")
        return 1-2*M/r

    def dAdr(t, r, theta, phi):
        M = metric.get_constant("M")
        return 2*M/r**2

    metric = pygro.Metric(
        name = name,
        line_element = line_element,
        coordinates = coordinates,
        A = A,
        dAdr = dAdr,
        transform = transform
    )

    metric.add_parameter("M", 1)

.. note::
    Notice how we have made use of the :meth:`~pygro.metric_engine.Metric.get_constant` method to access the `M` parameter inside the metric. In particular, since now the symbolic expression of the line element does not contain any :math:`M`, we had to manually add this parameter to the metric bi using the :meth:`~pygro.metric_engine.Metric.add_parameter` method. Using this approach, now we can link symbolic parameter of the metric to ones that we need to access from the auxiliary functions.