import sympy as sp
import numpy as np
import numpy.typing as npt
import json
import logging
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.core.function import AppliedUndef
from collections.abc import Iterable
from typing import Callable, Optional

diag = sp.diag

def parse_expr(expr):
    return sp.parse_expr(expr)

class Metric():
    r"""This is the main symbolic tool within PyGRO to perform tensorial calculations
    starting from the space-time metric. PyGRO uses the signature convention :math:`(-,\,+,\,+,\,+)`.
    
    After successful initialization (:py:func:`Metric.__init__`), the :py:class:`.Metric` instance has the following attributes:

    :ivar g: The symbolic representation of the :math:`4\times4` metric tensor.
    :vartype g: sympy.Matrix
    :ivar g_inv: The symbolic representation of the :math:`4\times4` *inverse* metric tensor.
    :vartype g_inv: sympy.Matrix
    :ivar x: The symbolic representation of the space-time coordinates.
    :vartype x: list[sympy.Symbol]
    :ivar dx: The symbolic representation of the space-time coordinate differential.
    :vartype dx: list[sympy.Symbol]
    :ivar u: The symbolic representation of the components 4-velocity (derivatives of the coordinates w.r.t. an affine parameter).
    :vartype u: list[sympy.Symbol]
    :ivar eq_u: The computed geodesic equations, *i.e.* the right-hand-side of the equation :math:`\ddot{x}^\mu = -\Gamma^{\mu}_{\nu\rho}\dot{x}^\nu\dot{x}^\rho` where :math:`\Gamma^{\mu}_{\nu\rho}` are the :py:func:`Metric.Christoffel` symbols of the metric.
    :vartype eq_u: list[sympy.Basic]
    :ivar u[i]_null: The expression of the *i*-th component of the 4-velocity of a null test particle as a function of the others, which normalizes the 4-velocity to :math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = 0`
    :vartype u[i]_null: sympy.Basic
    :ivar u[i]_timelike: The expression of the *i*-th component of the 4-velocity of a massive test particle as a function of the others, which normalizes the 4-velocity to :math:`g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu = -1`
    :vartype u[i]_timelike: sympy.Basic
    :ivar transform_functions: The expressions transform functions to pseudo-cartesian coordinates that can be either passed as arguments of the :py:class:`Metric` constructor or set with the :py:func:`Metric.set_coordinate_transformation` method.
    :vartype transform_functions: list[sympy.Basic]
    """
    instances = []
    
    def __init__(self,
        name: str,
        coordinates: list[str],
        line_element: Optional[str] = None,
        tensor: Optional[sp.Matrix] = None,
        transform: Optional[list[str]] = None,
        load: Optional[str] = None,
        **params
    ):
        """
            To instantiate a new ``Metric`` object call
            
            >>> spacetime_metric = pygro.Metric(name, coordinates, line_element)
            
            or
            
            >>> spacetime_metric = pygro.Metric(name, coordinates, tensor)
            
            with:

            :param name: The name of the metric to initialize.
            :type name: str
            :param coordinates: Four-dimensional list containing the symbolic expression of the space-time coordinates in which the ``line_element`` argument is written.
            :type coordinates: list[str]
            :param line_element: A string containing the symbolic expression of the line element (that will be parsed using ``sympy``) expressed in the space-time coordinates defined in the  `coordinates` argument.
            :type line_element: Optional[str]
            :param tensor: A sympy matrix containing the symbolic expression of the matrix rappresentation of the metric tensor in the given space-time coordinets.
            :type tensor: Optional[sp.Matrix]
            :param transform: Symbolic expressions containing the transformations between the chosen system of coordinates and a pseud-cartesian coordinate system, useful to :doc:`visualize` using the :py:func:`Metric.transform` function.
            :type transform: list[str]
            
            see :doc:`create_metric` for a detailed documentation on how to correctly define a space-time metric.
        """
        self._initialized = False
        self._initialized_metric = False
        self._geodesic_engine_linked = False

        if load:
            self.load_metric(load)
        
        self._initialize_metric(
            name, coordinates, line_element, tensor, transform, **params
        )
        
        Metric.instances.append(self)
    
    def _initialize_metric(self, name: str,
        coordinates: list[str],
        line_element: Optional[str] = None,
        tensor: Optional[sp.Basic] = None,
        transform_functions: Optional[list[str]] = None, **params):
        """Initializes the `Metric`. 

        :param \**kwargs:
            Required parameters are ``name``, ``coordinates`` and one between ``line_element`` or ``tensor``,
            but additional parameters can and *must* be passed on occasions (see :doc:`create_metric` to see examples). 
        """
        
        if (line_element is None) == (tensor is None):
            raise ValueError('initialize_metric needs at least "name", "coordinates", and one between "line_element" and "tensors" as arguments.')
        
        if line_element:
            if not isinstance(line_element, str):
                raise TypeError("line_element must be a string")
        if tensor:
            if not isinstance(tensor, sp.Matrix):
                raise TypeError("tensor must be a sympy.Matrix")
        
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")
        if not all([isinstance(coordinate, str) for coordinate in coordinates]):
            raise TypeError("'coordinates' must be a list of strings.")
        if len(coordinates) != 4:
            raise ValueError("'coordinates' must have dimension 4.")
        
        self.x = []
        self.dx = []
        self.x_str = []
        self.u = []

        self.name = name
        coordinates_input = coordinates
        
        logging.info(f"Initializing {self.name}.")
        
        for coordinate in coordinates_input:
            self.x.append(sp.symbols(coordinate))
            self.x_str.append(coordinate)

            velocity = "u_" + coordinate
            self.u.append(sp.symbols(velocity))

            differential = "d" + coordinate
            self.dx.append(sp.symbols(differential))
        
        if line_element:
            self.g = sp.zeros(4, 4)
            self.g_str = np.zeros((4, 4), dtype=object)
                    
            try:
                ds2_str = line_element
                ds2_sym = sp.expand(sp.parse_expr(ds2_str))
            except:
                raise ValueError("Please insert a valid expression for the line element.")
            else:
                for i, dx1 in enumerate(self.dx):
                    for j, dx2 in enumerate(self.dx):
                        self.g[i,j] = ds2_sym.coeff(dx1*dx2,1)
                        self.g_str[i,j] = str(self.g[i,j])

        if tensor:
            self.g_str = np.zeros((4, 4), dtype=str)
            if isinstance(tensor, sp.Matrix) and tensor.shape == (4,4):
                self.g = tensor
                for i in range(4):
                    for j in range(4):
                        self.g_str[i,j] = str(self.g[i,j])
            else:
                raise ValueError("Please insert a valid expression for the metric tensor.")
                
        logging.info("Calculating inverse metric.")
        self.g_inv = self.g.inv()
        self.g_inv_str = np.zeros([4,4], dtype = object)
        
        for i in range(4):
            for j in range(4):
                self.g_inv_str[i,j] = str(self.g_inv[i,j])
        
        logging.info("Calculating symbolic equations of motion.")
        self.eq_u = []
        self.eq_u_str = np.zeros(4, dtype = object)

        self.eq_x = []
        self.eq_x_str = np.zeros(4, dtype = object)

        for rho in range(4):
            eq = 0
            for mu in range(4):
                for nu in range(4):
                    eq += -self.Christoffel(mu, nu, rho)*self.u[mu]*self.u[nu]
            self.eq_u.append(eq)
            self.eq_u_str[rho] = str(eq)

            self.eq_x.append(self.u[rho])
            self.eq_x_str[rho] = str(self.u[rho])

        logging.info("Computing helper functions to normalize 4-velocity.")

        eq = 0

        for mu in range(4):
            for nu in range(4):
                eq += self.g[mu, nu]*self.u[mu]*self.u[nu]

        self.u0_s_null = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]
        self.u1_s_null = sp.solve(eq, self.u[1], simplify=False, rational=False)[0]
        self.u2_s_null = sp.solve(eq, self.u[2], simplify=False, rational=False)[0]
        self.u3_s_null = sp.solve(eq, self.u[3], simplify=False, rational=False)[0]

        eq += 1

        self.u0_s_timelike = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]
        self.u1_s_timelike = sp.solve(eq, self.u[1], simplify=False, rational=False)[0]
        self.u2_s_timelike = sp.solve(eq, self.u[2], simplify=False, rational=False)[0]
        self.u3_s_timelike = sp.solve(eq, self.u[3], simplify=False, rational=False)[0]
        
        self._initialized = True
        self.parameters = {}
        self.transform_s = []
        self.transform_functions = []
        self.transform_functions_str = []
        
        free_sym = list(self.g.free_symbols-set(self.x))
        free_func = list(self.g.atoms(AppliedUndef))

        if len(free_sym) > 0:
            for sym in free_sym:
                self.add_parameter(str(sym))
                if str(sym) in params:
                    self.set_constant(**{str(sym): params.get(str(sym))})
                else:
                    value = float(input(f"Insert value for {str(sym)}: "))
                    self.set_constant(**{str(sym): value})

        if len(free_func) > 0:
            for func in free_func:
                self.add_parameter(str(func))
                
                if not str(func.func) in params:
                    raise ValueError("Auxiliary functions shoudld be passed as arguments, either as strings (expression mode) or as python methods (functional mode).")
                
                if isinstance(params[str(func.func)], str):
                    kind = "expr"
                elif callable(params[str(func.func)]):
                    kind = "py"

                if kind == "expr":
                    self.parameters[str(func)]['kind'] = "expression"
                    expr_str = params[str(func.func)]

                    expr_sym = sp.expand(sp.parse_expr(expr_str))

                    self.parameters[str(func)]['value'] = expr_sym

                    expr_free_sym = expr_sym.free_symbols-set([self.parameters[parameter]["symbolic"] for parameter in self.parameters])-set(self.x)

                    for sym in expr_free_sym:
                        self.add_parameter(str(sym))
                        if str(sym) in params:
                            self.set_constant(**{str(sym): params.get(str(sym))})
                        else:
                            value = float(input("Insert value for {}: ".format(str(sym))))
                            self.set_constant(**{str(sym): value})

                    for arg in list(func.args):
                        self.parameters[str(func)][f"d{str(func.func)}d{str(arg)}"] = expr_sym.diff(arg)

                elif kind == "py":

                    self.parameters[str(func)]['kind'] = "pyfunc"
                    
                    args = list(func.args)
                    derlist = [f"d{str(func.func)}d{str(arg)}" for arg in args]

                    arg_derlist = {}

                    for der in derlist:
                        if not der in params:
                            raise ValueError(f"Functional derivatives of auxiliary functions are required as extra-arguments. In this case '{der}' is missing.")
                        else:
                            arg_derlist[der] = params[der]

                    self.set_function_to_parameter(str(func), params[str(func.func)], **arg_derlist)

        if transform_functions:
            transform = transform_functions
            if len(transform) != 4:
                raise ValueError('"transform" should be a 4-dimensional list of strings')
            
            for i, x in enumerate(["t", "x", "y", "z"]):
                try:
                    x_inpt = transform[i]
                    x_symb = sp.parse_expr(x_inpt)
                except:
                    raise ValueError(f"Insert a valid expression for transform function {x}")
                else:
                    self.transform_s.append(x_symb)
                    self.transform_functions_str.append(x_inpt)
                    self.transform_functions.append(sp.lambdify([*self.x, *self.get_parameters_symb()], x_symb, 'numpy'))
                    
        self._g_ff = lambdify([*self.x, *self.get_parameters_symb()], self.subs_functions(self.g))
        
        logging.info(f"The Metric ({self.name}) has been initialized.")

    def save_metric(self, filename):
        r"""Saves the metric into a *.metric* file which can later be loaded with the :py:func:`Metric.load_metric` method.

        :param filename: The name of the *.metric* file in which to save the metric.
        :type filename: str
        """
        if self._initialized:
            with open(filename, "w+") as file:
                output = {}
                
                output['name'] = self.name
                output['g'] = self.g_str.tolist()
                output['x'] = self.x_str
                output['g_inv'] = self.g_inv_str.tolist()
                output['eq_x'] = self.eq_x_str.tolist()
                output['eq_u'] = self.eq_u_str.tolist()
                output['u0_timelike'] = str(self.u0_s_timelike)
                output['u1_timelike'] = str(self.u1_s_timelike)
                output['u2_timelike'] = str(self.u2_s_timelike)
                output['u3_timelike'] = str(self.u3_s_timelike)
                output['u0_null'] = str(self.u0_s_null)
                output['u1_null'] = str(self.u1_s_null)
                output['u2_null'] = str(self.u2_s_null)
                output['u3_null'] = str(self.u3_s_null)

                output["expressions"] = []

                expressions = self.get_parameters_expressions()

                for f in expressions:
                    output["expressions"].append({'name': expressions[f]['symbol'], 'value': str(expressions[f]['value'])})
                
                output["functions"] = []

                functions = self.get_parameters_functions()

                for f in functions:
                    output["functions"].append({'name': functions[f]['symbol']})
                
                if(self.transform_functions):
                    output['transform'] = self.transform_functions_str
                
                json.dump(output, file)

        else:
            logging.info("Initialize or load a metric before saving.")
    
    def load_metric(self, filename, **params):
        r"""Loads the metric from a *.metric* file which has been saved through the :py:func:`Metric.save_metric` method.

        :param filename: The name of the *.metric* file from which to load the metric.
        :type filename: str
        """
        f = open(filename, "r")

        load = json.load(f)

        self._load_metric_from_json(load, **params)

        
    def _load_metric_from_json(self, metric_json, **params):

        self.json = metric_json
        load = metric_json

        self.name = load['name']

        if not "expressions" in load:
            load["expressions"] = []
        
        if not "functions" in load:
            load["functions"] = []

        logging.info("Loading {}".format(self.name))
        
        self.x = []
        self.dx = []
        self.x_str = []
        self.u = []

        for i in range(4):

            coordinate = load['x'][i]
            self.x.append(sp.symbols(coordinate))
            self.x_str.append(coordinate)

            velocity = "u_" + coordinate
            self.u.append(sp.symbols(velocity))

            differential = "d" + coordinate
            self.dx.append(sp.symbols(differential))
        
        self.g = sp.zeros(4, 4)
        self.g_inv = sp.zeros(4, 4)
        self.eq_u = []
        self.eq_x = []
        
        self.g_str = np.zeros([4,4], dtype = object)
        self.g_inv_str = np.zeros([4,4], dtype = object)
        self.eq_u_str = np.zeros(4, dtype = object)
        self.eq_x_str = np.zeros(4, dtype = object)

        for i in range(4):
            for j in range(4):

                component = load['g'][i][j]
                self.g[i,j] = sp.parse_expr(component)
                self.g_str[i,j] = component
                component = load['g_inv'][i][j]
                self.g_inv[i,j] = sp.parse_expr(component)
                self.g_inv_str[i,j] = component

            self.eq_u.append(sp.parse_expr(load['eq_u'][i]))
            self.eq_x.append(sp.parse_expr(load['eq_x'][i]))
            self.eq_u_str[i] = load['eq_u'][i]
            self.eq_x_str[i] = load['eq_x'][i]
        
        self.u0_s_null = sp.parse_expr(load['u0_null'])
        self.u1_s_null = sp.parse_expr(load['u1_null'])
        self.u2_s_null = sp.parse_expr(load['u2_null'])
        self.u3_s_null = sp.parse_expr(load['u3_null'])
        
        self.u0_s_timelike = sp.parse_expr(load['u0_timelike'])
        self.u1_s_timelike = sp.parse_expr(load['u1_timelike'])
        self.u2_s_timelike = sp.parse_expr(load['u2_timelike'])
        self.u3_s_timelike = sp.parse_expr(load['u3_timelike'])
        
        self.parameters = {}
        self.transform_functions = []
        self.transform_s = []

        free_sym = list(self.g.free_symbols-set(self.x))
        
        self._initialized = True

        if(len(free_sym)) > 0:
            for sym in free_sym:
                self.add_parameter(str(sym))
                if str(sym) in params:
                    self.set_constant(**{str(sym): params.get(str(sym))})
                else:
                    value = float(input("Insert value for {}: ".format(str(sym))))
                    self.set_constant(**{str(sym): value})
        
        for expression in load["expressions"]:
            self.add_parameter(expression['name'])

            self.parameters[expression['name']]['kind'] = "expression"

            func_sym = sp.parse_expr(expression['name'])

            expr_str = expression['value']
            expr_sym = sp.expand(sp.parse_expr(expr_str))

            self.parameters[expression['name']]['value'] = expr_sym

            expr_free_sym = expr_sym.free_symbols-set([self.parameters[parameter]["symbolic"] for parameter in self.parameters])-set(self.x)

            for sym in expr_free_sym:
                self.add_parameter(str(sym))
                if str(sym) in params:
                    self.set_constant(**{str(sym): params.get(str(sym))})
                else:
                    value = float(input("Insert value for {}: ".format(str(sym))))
                    self.set_constant(**{str(sym): value})

            for arg in list(func_sym.args):
                self.parameters[expression['name']][f"d{str(func_sym.func)}d{str(arg)}"] = expr_sym.diff(arg)

        for function in load["functions"]:

            self.add_parameter(function['name'])
            self.parameters[function['name']]['kind'] = "pyfunc"

            func_sym = sp.parse_expr(function['name'])

            func = str(func_sym.func)
            args = list(func_sym.args)

            if str(func) in params:
                self.parameters[function['name']]['value'] = params[str(func)]
            else:
                logging.warning(f"Remember to set the Python function for {str(func)}.")
            
            for arg in args:
                dername = f"d{func}d{str(arg)}"

                if dername in params:
                    self.parameters[function['name']][dername] = params[dername]
                else:
                    logging.warning(f"Remember to set the Python function for {dername}.")

        self._initialized_metric = True
        
        if "transform" in load:
            for i in range(4):
                transf = load['transform'][i]
                transform_function = sp.parse_expr(transf)
                self.transform_s.append(transform_function)
                self.transform_functions.append(sp.lambdify([*self.x, *self.get_parameters_symb()], transform_function, 'numpy'))

                
        self._g_ff = lambdify([*self.x, *self.get_parameters_symb()], self.subs_functions(self.g))

        logging.info(f"The Metric ({self.name}) has been initialized.")

    def add_parameter(self, symbol: str, value: float = None):
        r"""Manually adds a constant parameter to the `Metric` object. The value of this parameter must be specified
        
        :param symbol: The symbolic name of the parameter.
        :type symbol: str
        :param value: The value of the parameter.
        :type value: float
        """
        if self._initialized:
            self.parameters[symbol] = {}
            self.parameters[symbol]['symbol'] = symbol
            self.parameters[symbol]['symbolic'] = sp.parse_expr(symbol)
            self.parameters[symbol]['value'] = value
            if value != None:
                self.parameters[symbol]['kind'] = "constant"
            else:
                self.parameters[symbol]['kind'] = None
        else:
            logging.error("Initialize or load a metric before adding parameters.")
    
    def set_constant(self, **params):
        r"""Sets the value of a given parameter in the `Metric`. Useful to change the value of an existing paramter.
        The parameter and its value to be set should be passed as a keyword argument to the `set_constant` function.
        
        :Example:
        >>> metric.set_constant(M = 1)
        """
        if self._initialized:
            for param in params:
                try:
                    self.parameters[str(param)]['value'] = params[param]
                    self.parameters[str(param)]['kind'] = "constant"
                except:
                    logging.error(f"No parameter named '{str(param)}' in the Metric.")
                    break
        else:
            logging.error("Initialize or load a metric before adding parameters.")
    
    def set_expression_to_parameter(self, param, expr_str):
        r"""Selects the ``param`` element from the ``metric.parameters`` dictionary, sets its kind to ``"expr"`` and assignes to it a value which is the sybolic parsed
        expression in the `expr_str` argument.
        
        :param param: The symbolic name of the parameter to modify.
        :type param: str
        :param expr_str: The symbolic expresison to assign to this parameter.
        :type expr_str: str
        """
        if self._initialized:
            if param in self.parameters:
                self.parameters[str(param)]['value'] = sp.parse_expr(expr_str)
                self.parameters[str(param)]['kind'] = "expression"
                func = self.parameters[str(param)]['symbolic']

                expr_sym = sp.expand(sp.parse_expr(expr_str))

                self.parameters[str(param)]['value'] = expr_sym

                expr_free_sym = expr_sym.free_symbols-set([self.parameters[parameter]["symbolic"] for parameter in self.parameters])-set(self.x)

                for sym in expr_free_sym:
                    self.add_parameter(str(sym))
                    value = float(input("Insert value for {}: ".format(str(sym))))
                    self.set_constant(**{str(sym): value})


                for arg in list(func.args):
                    self.parameters[str(param)][f"d{str(func.func)}d{str(arg)}"] = expr_sym.diff(arg)

            else:
                raise TypeError(f"No parameter named '{str(param)}' in the Metric.")
        else:
            logging.warning("Initialize or load a metric before adding parameters.")


    def set_function_to_parameter(self, param: str, function: Callable, **derivatives: list[Callable]):
        r"""Selects the ``param`` element from the ``metric.parameters`` dictionary, sets its kind to ``"py"`` and assignes as its value the ``function`` method that is passed as argument. 

        .. note::
            The ``function`` argument **must** be a method which admits four floats and returns a single float as results.
            However, in the ``line_element`` argument, upon initialization of the ``Metric``, only explicit coordinate dependences of the
            functional parameter must be indicated. Moreover, for each of the explicit coordinate dependence of the function its derivative should also
            be passed in argument (see :doc:`create_metric` for a tutorial on how to do this).

        :Example:    
        
        .. code-block::

            line_element = "-A(r)*dt**2+1/A(r)*dr**2+(r**2)*(dtheta**2+sin(theta)**2*dphi**2)"
            
            def A(t, r, theta, phi):
                M = metric.get_constant("M")
                return 1-2*M/r

            def dAdr(t, r, theta, phi):
                M = metric.get_constant("M")
                return 2*M/r**2
                
            metric = pygro.Metric(..., A = A, dAdr = dAdr)
            
        
        :param param: The symbolic name of the parameter to modify.
        :type param: str
        :param function: The method to be called as the auxiliary function in the ``Metric``.
        :type function: Callable
        :param derivatives: methods to be called as the derivatives of the auxiliary function in the ``Metric`` with respect to the variables on which it depends. The argument keyword must be d[function]d[variable]. For example, if ``A(r, theta)`` is the auxiliary function, the derivatives ``(..., dAdr = [...], dAdtheta = [...])`` must be provided.
        :type derivatives: list[Callable]
        """
        if self._initialized:
            
            args = list(self.parameters[str(param)]['symbolic'].args)
            func = self.parameters[str(param)]['symbolic'].func
            self.parameters[str(param)]['kind'] = "pyfunc"

            self.parameters[str(param)]['value'] = implemented_function(f"{func}_func", function)

            for arg in args:
                der = f"d{func}d{arg}"
                if not der in derivatives:
                    raise KeyError(f"Specify a meethod for the derivative of the function with respect to {arg}")
                else:
                    self.parameters[str(param)][der] = implemented_function(f"d{func}d{arg}_func", derivatives[der])
        else:
            logging.info("Initialize or load a metric before adding parameters.")
            
    def get_parameters_symb(self):
        r"""
        :Returns:
        The symbolic representation of all the elements in ``pygro.Metric.parameters``.
        """
        return [self.parameters[constant]['symbolic'] for constant in self.parameters if self.parameters[constant]['kind'] == "constant"]

    def get_parameters_val(self):
        r"""
        :Returns:
        The numerical value of all the elements in ``pygro.Metric.parameters`` of type ``"constant"``.
        """
        return [self.parameters[constant]['value'] for constant in self.parameters if self.parameters[constant]['kind'] == "constant"]

    def get_parameters_constants(self):
        r"""
        :Returns:
        All the elements in ``pygro.Metric.parameters`` of type ``"constant"``.
        """
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "constant"}
    
    def get_parameters_expressions(self):
        r"""
        :Returns:
        All the elements in ``pygro.Metric.parameters`` of type ``"epxression"`` (auxiliary expressions).
        """
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "expression"}

    def get_parameters_functions(self):
        r"""
        :Returns:
        All the elements in ``pygro.Metric.parameters`` of type ``"pyfunc"`` (auxiliary functions).
        """
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "pyfunc"}
    
    def get_constant(self, param: str):
        """Returns the value of the constant ``parameter``, if defined.
        
        :param param: The symbolic name of the parameter to retrieve.
        :type param: str
        
        :Raises:
            ValueError: if ``parameter`` not defined.
        """
        if self.parameters[param]["kind"] == "constant":
            return self.parameters[param]["value"]
        else:
            raise TypeError(f"Unknown constant {param}")

    # Auxiliary functions

    def subs_functions(self, expr: sp.Basic):
        """:Returns: 
        The input ``sympy`` symbolic expression ``expr`` where the all the occurences of symbolic expressions have beem substituted by their representation.
        
        :param expr: The ``sympy`` symbolic expression in which to substitute funcitons.
        :type expr: sp.Basic
        """
        functions = self.get_parameters_expressions()

        if len(functions) > 0:
            for func in functions:

                subs = []
                f = functions[func]['symbolic']

                for arg in f.args:
                    subs.append((sp.Derivative(f, arg), functions[func][f"d{f.func}d{arg}"]))
                
                expr = expr.subs(subs)
            
                expr = expr.subs(f, functions[func]['value'])
        
        functions = self.get_parameters_functions()

        if len(functions) > 0:
            for func in functions:

                subs = []
                f = functions[func]['symbolic']

                for arg in f.args:
                    subs.append((sp.Derivative(f, arg), functions[func][f"d{f.func}d{arg}"](*self.x)))
                
                expr = expr.subs(subs)
            
                expr = expr.subs(f, functions[func]['value'](*self.x))
        
        return expr


    def evaluate_parameters(self, expr: sp.Basic):
        """:Returns: 
        The input ``sympy`` symbolic expression ``expr`` where the all the constant parameters have been evalueated with their corrispodning values.
        
        :param expr: The ``sympy`` symbolic expression in which to substitute funcitons.
        :type expr: sp.Basic
        
        :Raises:
            ValueError: if the value of the parameter has not been previously set.
        """
        if any(x['value'] == None for x in self.parameters.values()):
            raise ValueError("You must set_constant for every constant value in your metric, before evaluating.")
        else:
            subs = []
            const = self.get_parameters_constants()
            for c in const:
                x = const[c]['symbol']
                y = const[c]['value']
                subs.append([x,y])
            return self.subs_functions(expr).subs(subs)
    
    def get_evaluable_function(self, expr: str | sp.Basic):
        """
        :Returns: 
        A python callable from a sympy expressions, where all auxiliary functions and parameters from the metric have been evaluated.

        :param expr: The ``sympy`` symbolic expression to evaluate or a string containing the symbolic expression.
        :type expr: str | sp.Basic
        """
        if isinstance(expr, str):
            expr = sp.parse_expr(expr)
        
        free_sym = list(expr.free_symbols-set(self.get_parameters_symb()))
        
        free_sym_ordered = [x_u for x_u in [*self.x, *self.u] if x_u in free_sym]
        
        return sp.lambdify(free_sym_ordered, self.evaluate_parameters(self.subs_functions(expr)))
        

    def set_coordinate_transformation(self, transform_functions: list[str]):
            """Sets the coordinate transforamtion to a pseduo-cartesian system of coordinates.
            
            :Example:
            
            .. code-block::

                transform_functions = [
                    "t",
                    "r*sin(theta)*cos(phi)",
                    "r*sin(theta)*sin(phi)",
                    "r*cos(theta)"
                ]
                
                metric.set_coordinate_transformation(transform_functions = transform_functions)


            :param transform_functions: A list of strings containing the symbolic expressions of the transformation functions
            :type transform_functions: list[str]
            """
            self.transform_s = []
            self.transform_functions = []
            for i, x in enumerate(["t", "x", "y", "z"]):
                try:
                    x_symb = sp.parse_expr(transform_functions)
                except:
                    raise ValueError(f"Insert a valid expression for transformation to the cartesian coordinate: {x}")
                
                self.transform_s.append(x_symb)
                self.transform_functions.append(sp.lambdify([self.x], x_symb, 'numpy'))
    
    def transform(self, X: npt.ArrayLike):
        r"""A function to apply the transformation defined in ``transform_functions`` to the given point.
        
        :param X: the point(s) to transform. This can either be a 4-dimensional array or a :math:`4\times n` ``np.array`` (`i.e.` a collection of points).
        :type point: npt.ArrayLike
        
        :Returns:
        The transformed values of the coordinates of the point(s).
        """
        if self.transform_functions:
            return self.transform_functions[0](*X, *self.get_parameters_val()), self.transform_functions[1](*X, *self.get_parameters_val()), self.transform_functions[2](*X, *self.get_parameters_val()), self.transform_functions[3](*X, *self.get_parameters_val())
        else:
            raise TypeError("Coordinate transformations not set. Use .set_coordinate_transformation method in Metric class.")
        
    def _g_f(self, x):
        return self._g_ff(*x, *self.get_parameters_val())
    
    def parse_expr(self, expr: sp.Basic):
        return sp.parse_expr(expr)

    def Christoffel(self, mu: int, nu: int, rho: int):
        r"""The symbolic representation of the mu-nu-rho Christoffel symbol, :math:`\Gamma^{\mu}_{\nu\rho}` related to the metric tensor.
        """
        ch = 0
        for sigma in range(4):
            ch += self.g_inv[rho,sigma]*(self.g[sigma, nu].diff(self.x[mu])+self.g[mu, sigma].diff(self.x[nu])-self.g[mu, nu].diff(self.x[sigma]))/2
        return ch
    
    def Lagrangian(self):
        """The symbolic representation of the test particle Lagrangian related to the metric tensor.
        """
        lagrangian = 0

        for mu in range(4):
            for nu in range(4):
                lagrangian += self.g[mu,nu]*self.u[mu]*self.u[nu]/2
        
        return lagrangian
    
    def norm(self, point: Iterable[float], vec: Iterable[float]) -> float:
        """
        :Returns:
        The norm of the 4-vector ``vec`` computed at the given ``point`` in space-time. Uses the currently set values of the constant ``parameters``.
        
        :param point: 4-dimensional array of numbers representing the position in space-time at which to compute the norm of the 4-vector.
        :type point: Iterbale[float]
        :param vec: 4-dimensional array of numbers representing the components of the 4-vector.
        :type vec: Iterbale[float]
        
        :Raises:
            ValueError: if the input arrays are not 4-dimensional.
        """
        if (len(point) != 4) or (len(vec) != 4):
            raise ValueError("The 'point' and the 'vector' must be 4-dimensional arrays.")
        n = 0
        for mu in range(4):
            for nu in range(4):
                n += self.g_f(point)[mu,nu]*vec[mu]*vec[nu]
        return n