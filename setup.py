 from setuptools import setup

 setup(
   name='PIGRO',
   version='0.0.1',
   author='Riccardo Della Monica',
   author_email='dellamonicariccardo@gmail.com',
   packages=['pigro'],
   url='http://pypi.python.org/pypi/PIGRO/',
   license='LICENSE.txt',
   description='A Python Integrator for General Relativistic Orbits',
   long_description=open('README.md').read(),
   install_requires=[
       "sympy",
       "numpy",
       "tqdm",
       "pickle"
   ],
)