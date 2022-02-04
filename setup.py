from setuptools import setup, find_packages

setup(
  name='PyGRO',
  version='0.0.10',
  author='Riccardo Della Monica',
  author_email='dellamonicariccardo@gmail.com',
  packages=find_packages(),
  license='LICENSE.txt',
  description='A Python Integrator for General Relativistic Orbits',
  long_description=open('README.md').read(),
  long_description_content_type='text/markdown',
  install_requires=[
    "sympy",
    "numpy",
    "scipy",
    "cython"
  ],
  keywords=['python', 'first package'],
  classifiers= [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "Programming Language :: Python :: 3",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
  ]
)