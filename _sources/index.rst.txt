.. image:: _static/pygro-banner.png
  :width: 100%
  :alt: PyGRO
  :class: only-light

.. image:: _static/pygro-banner-dark.png
  :width: 100%
  :alt: PyGRO
  :class: only-dark

PyGRO [pronounced /ˈpiɡro/, from the italian **pigro**: *lazy, indolent*] is a Python library that provides a series of tools to perform the numerical integration of the geodesic equations describing a particle or photon orbit in any metric theory of gravity, given an analytic expression of the metric tensor.

Documentation
-----------------

This documentation comes with a series of tutorials that will guide you through the different possibilities that PyGRO offers

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   getting_started
   create_metric
   integrate_geodesic
   define_observer
   integrating_orbits
   visualize
   integrators

Moreover in the example notebooks we investigate some specific situations:
   
.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks:

   examples/Schwarzschild-Precession.ipynb
   examples/Schwarzschild-PhotonOrbit.ipynb
   examples/Schwarzschild-ISCO.ipynb
   examples/Wormhole.ipynb
   examples/Kerr-Photons.ipynb
   examples/S2_MCMC

Finally we provide the users with a detailed API guide that offers support for all the classes and methods in PyGRO

.. toctree::
   :maxdepth: 2
   :caption: API:

   metricengine
   geodesicengine
   geodesic
   observer
   orbit
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
