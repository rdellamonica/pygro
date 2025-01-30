Getting started
===============

Installation
----------------

PyGRO is distributed as a Python package that can be installed through the PyPi package manager via:

.. code-block:: console

   pip install pygro

Alternatively, you can clone PyGRO's Github repo and install it manually:

.. code-block:: console

   git clone https://github.com/rdellamonica/pygro.git
   cd pygro
   python -m pip install .

Requirements
----------------

To successfully use this package, ensure you have the following dependencies installed. These dependencies are specified in the ``requirements.txt`` file and can be installed easily using ``pip``. Below are the exact versions required for compatibility:

- **numpy>=2.2.2**
- **scipy>=1.15.1**
- **sympy>=1.11.1**

You can install these dependencies by running the following command in your terminal:

.. code-block:: console
   
   pip install -r requirements.txt

Ensure your Python environment is properly set up and compatible with the above versions to avoid compatibility issues.

License
----------------

PuGRO is licensed under the GNU General Public License (GPL), which ensures that the software remains free and open source. Under the terms of the GPL:

1. **Freedom to Use**: You can use the software for any purpose, whether personal, educational, or commercial.  
2. **Freedom to Modify**: You can study the source code, make changes, and adapt the software to meet your needs.  
3. **Freedom to Share**: You can redistribute copies of the original software to others.  
4. **Freedom to Share Modifications**: You can distribute modified versions of the software, provided you also release them under the GPL.  

**Conditions**:  
- If you distribute the software (modified or unmodified), you must include the full source code or make it available, along with a copy of the GPL license.  
- Any modifications or derivative works you distribute must also be licensed under the GPL, ensuring they remain free and open source.  
- The software comes with no warranty, to the extent permitted by law.

For more details, visit https://www.gnu.org/licenses/.

Citing PyGRO
--------------------

PyGRO will be accompanied by a scientific publication.

Please cite it properly attribute it in your works:

.. code-block:: latex

   @ARTICLE{PyGRO2025,
         author = {{Della Monica}, Riccardo},
          title = "{PyGRO: a Python integrator for General Relativistic Orbits}",
        journal = {arXiv e-prints},
       keywords = {General Relativity and Quantum Cosmology},
           year = 2025,
          month = feb,
         eprint = {arxiv:2502.xxxx},
   primaryClass = {gr-qc},
   }

   

