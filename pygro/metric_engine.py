import sympy as sp
import numpy as np
import json
import logging
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.core.function import AppliedUndef

diag = sp.diag

def parse_expr(expr):
    return sp.parse_expr(expr)

class Metric():
    r"""This is the main symbolic tool within PyGRO to perform tensorial calculations
    starting from the spacetime metric. To instantiate a new `Metric` object call
    
    >>> spacetime_metric = pygro.Metric(**kwargs)
    
    with accepted ``**kwargs`` being:

    :param name: The name of the metric to initialize.
    :type name: str
    :param coordinates: Four-dimensional list containing the symbolic expression of the space-time coordinates in which the `line_element` argument is written.
    :type coordinates: list of str
    :param line_element: A string containing the symbolic expression of the line element (that will be parsed using `sympy`) expressed in the space-time coordinates defined in the 
    `coordinates` argument.
    :type line_element: str
    :param matrix: A sympy matrix containing the symbolic expression of the matrix rappresentation of the metric tensor in the given space-time coordinets.
    :type line_element: str

    
    After successful initialization, the ``Metric`` instance has the following attributes:

    :ivar g: The symbolic representation of the metric tensor. It is a :math:`4\times4` ``sympy.Matrix`` object.
    :type g: sympy.Matric

    """
    instances = []
    MINIMAL_KWARGS = ["name", "coordinates"]
    METRIC_KWARGS = ["line_element", "tensor"]
    
    def __init__(self, **kwargs):
        self.initialized = False
        self.initialized_metric = False
        self.geodesic_engine_linked = False

        if len(kwargs) > 0:
            if "load" in kwargs:
                self.load_metric(kwargs["load"])
            self.initialize_metric(**kwargs)
        
        Metric.instances.append(self)
    
    def initialize_metric(self, **kwargs):
        """Initializes the `Metric`. 

        :param \**kwargs:
            Required parameters are ``name``, ``coordinates`` and one between ``line_element`` or ``tensor``,
            but additional parameters can and *must* be passed on occasions (see :doc:`create_metric` to see examples). 
        """

        self.x = []
        self.dx = []
        self.x_str = []
        self.u = []

        minimal_arguments_check = True
        for kwarg in Metric.MINIMAL_KWARGS:
            minimal_arguments_check &= kwarg in kwargs
        minimal_arguments_check &= any([kwarg in Metric.METRIC_KWARGS for kwarg in kwargs])
        
        if not minimal_arguments_check:
            raise ValueError('initialize_metric needs at least "name", "coordinates", and one between "line_element" and "tensors" as arguments.')
        

        self.name = kwargs['name']
        coordinates_input = kwargs["coordinates"]
        
        logging.info(f"Initializing {self.name}.")
        
        if len(coordinates_input) != 4:
            raise ValueError('coordinates should be a 4-dimensional list of strings')
        
        for coordinate in coordinates_input:
            self.x.append(sp.symbols(coordinate))
            self.x_str.append(coordinate)

            velocity = "u_" + coordinate
            self.u.append(sp.symbols(velocity))

            differential = "d" + coordinate
            self.dx.append(sp.symbols(differential))
        
        if "line_element" in kwargs:
            self.g = sp.zeros(4, 4)
            self.g_str = np.zeros((4, 4), dtype=object)
                    
            try:
                ds2_str = kwargs["line_element"]
                ds2_sym = sp.expand(sp.parse_expr(ds2_str))
            except:
                raise ValueError("Please insert a valid expression for the line element.")
            else:
                for i, dx1 in enumerate(self.dx):
                    for j, dx2 in enumerate(self.dx):
                        self.g[i,j] = ds2_sym.coeff(dx1*dx2,1)
                        self.g_str[i,j] = str(self.g[i,j])

        elif "tensor" in kwargs:
            tensor = kwargs['tensor']
            self.g_str = np.zeros((4, 4), dtype=str)
            if isinstance(tensor, sp.Matrix) and tensor.shape == (4,4):
                self.g = kwargs['tensor']
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

        logging.info("Computing helper function to normalize 4-velocity.")

        eq = 0

        for mu in range(4):
            for nu in range(4):
                eq += self.g[mu, nu]*self.u[mu]*self.u[nu]

        self.u0_s_null = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]

        eq += 1

        self.u0_s_timelike = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]
            

        self.initialized = True
        self.parameters = {}
        self.transform_s = []
        self.transform_functions = []
        self.transform_functions_str = []
        
        free_sym = list(self.g.free_symbols-set(self.x))
        free_func = list(self.g.atoms(AppliedUndef))

        if len(free_sym) > 0:
            for sym in free_sym:
                self.add_parameter(str(sym))
                if str(sym) in kwargs:
                    self.set_constant(**{str(sym): kwargs.get(str(sym))})
                else:
                    value = float(input(f"Insert value for {str(sym)}: "))
                    self.set_constant(**{str(sym): value})

        if len(free_func) > 0:
            for func in free_func:
                self.add_parameter(str(func))
                
                if not str(func.func) in kwargs:
                    raise ValueError("Auxiliary functions shoudld be passed as arguments, either as strings (expression mode) or as python methods (functional mode).")
                
                if isinstance(kwargs[str(func.func)], str):
                    kind = "expr"
                elif callable(kwargs[str(func.func)]):
                    kind = "py"

                if kind == "expr":
                    self.parameters[str(func)]['kind'] = "expression"
                    expr_str = kwargs[str(func.func)]

                    expr_sym = sp.expand(sp.parse_expr(expr_str))

                    self.parameters[str(func)]['value'] = expr_sym

                    expr_free_sym = expr_sym.free_symbols-set([self.parameters[parameter]["symbolic"] for parameter in self.parameters])-set(self.x)

                    for sym in expr_free_sym:
                        self.add_parameter(str(sym))
                        if str(sym) in kwargs:
                            self.set_constant(**{str(sym): kwargs.get(str(sym))})
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
                        if not der in kwargs:
                            raise ValueError(f"Functional derivatives of auxiliary functions are required as extra-arguments. In this case '{der}' is missing.")
                        else:
                            arg_derlist[der] = kwargs[der]

                    self.set_function_to_parameter(str(func), kwargs[str(func.func)], **arg_derlist)

        if "transform" in kwargs:
            transform = kwargs["transform"]
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
                    self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(x_symb), 'numpy'))
                    
        self.g_ff = lambdify([*self.x, *self.get_parameters_symb()], self.subs_functions(self.g))
        
        logging.info(f"The Metric ({self.name}) has been initialized.")

    def save_metric(self, filename):
        r"""Saves the metric into a *.metric* file which can later be loaded with the :py:func:`Metric.load_metric` method.

        :param filename: The name of the *.metric* file in which to save the metric.
        :type filename: str
        """
        if self.initialized:
            with open(filename, "w+") as file:
                output = {}
                
                output['name'] = self.name
                output['g'] = self.g_str.tolist()
                output['x'] = self.x_str
                output['g_inv'] = self.g_inv_str.tolist()
                output['eq_x'] = self.eq_x_str.tolist()
                output['eq_u'] = self.eq_u_str.tolist()
                output['u0_timelike'] = str(self.u0_s_timelike)
                output['u0_null'] = str(self.u0_s_null)

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

        self.load_metric_from_json(load, **params)

        
    def load_metric_from_json(self, metric_json, **params):

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
        self.u0_s_timelike = sp.parse_expr(load['u0_timelike'])
        
        self.parameters = {}
        self.transform_functions = []
        self.transform_s = []

        free_sym = list(self.g.free_symbols-set(self.x))
        
        self.initialized = True

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

        self.initialized_metric = True
        
        if "transform" in load:
            for i in range(4):
                transf = load['transform'][i]
                transform_function = sp.parse_expr(transf)
                self.transform_s.append(transform_function)
                self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(transform_function), 'numpy'))

        logging.info(f"The Metric ({self.name}) has been initialized.")

    def add_parameter(self, symbol, value = None):
        if self.initialized:
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
        if self.initialized:
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
        if self.initialized:
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


    def set_function_to_parameter(self, param, function, **derivatives):
        r"""Selects the ``param`` element from the ``metric.parameters`` dictionary, sets its kind to ``"py"`` and assignes as its value
        the ``function`` method that is passed as argument. 

        .. note::
            The ``function`` argument **must** be a method which admits four floats and returns a single float as results.
            However, in the ``line_element`` argument, upon initialization of the ``Metric``, only explicit coordinate dependences of the
            functional parameter must be indicated. Moreover, for each of the explicit coordinate dependence of the function its derivative should also
            be passed in argument

            :Example:
        
        :param param: The symbolic name of the parameter to modify.
        :type param: str
        :param expr_str: The symbolic expresison to assign to this parameter.
        :type expr_str: str
        """
        if self.initialized:
            
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

    # Getters

    def get_parameters_symb(self):
        return [self.parameters[constant]['symbolic'] for constant in self.parameters if self.parameters[constant]['kind'] == "constant"]

    def get_parameters_val(self):
        return [self.parameters[constant]['value'] for constant in self.parameters if self.parameters[constant]['kind'] == "constant"]

    def get_parameters_constants(self):
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "constant"}
    
    def get_parameters_expressions(self):
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "expression"}

    def get_parameters_functions(self):
        return {self.parameters[constant]['symbol']: self.parameters[constant] for constant in self.parameters if self.parameters[constant]["kind"] == "pyfunc"}
    
    def get_constant(self, parameter):
        """Returns the value of the constant ``parameter``, if defined.
        """
        if self.parameters[parameter]["kind"] == "constant":
            return self.parameters[parameter]["value"]
        else:
            raise TypeError(f"Unknown constant {parameter}")

    # Auxiliary functions

    def subs_functions(self, expr):
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


    def evaluate_parameters(self, expr):
        if any(x['value'] == None for x in self.parameters.values()):
            logging.error("You must set_constant for every constant value in your metric, before evaluating.")
        else:
            subs = []
            const = self.get_parameters_constants()
            for c in const:
                x = const[c]['symbol']
                y = const[c]['value']
                subs.append([x,y])
            return self.subs_functions(expr).subs(subs)

    def set_coordinate_transformation(self, transform_functions):
            self.transform_s = []
            self.transform_functions = []
            for i, x in enumerate(["t", "x", "y", "z"]):
                try:
                    x_symb = sp.parse_expr(transform_functions)
                except:
                    raise ValueError(f"Insert a valid expression for transformation to the cartesian coordinate: {x}")
                
                self.transform_s.append(x_symb)
                self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(x_symb), 'numpy'))
    
    def parse_expr(self, expr):
        return sp.parse_expr(expr)

    def transform(self, X):
        if self.transform_functions:
            return self.transform_functions[0](X), self.transform_functions[1](X), self.transform_functions[2](X), self.transform_functions[3](X)
        else:
            raise TypeError("Coordinate transformations not set. Use .set_coordinate_transformation method in Metric class.")
        
    def g_f(self, x):
        return self.g_ff(*x, *self.get_parameters_val())

    def Christoffel(self, mu, nu, rho):
        """The mu-nu-rho Christoffel symbol, :math:`\Gamma^{\mu}_{\nu\rho}` related to the metric tensor.
        """
        ch = 0
        for sigma in range(4):
            ch += self.g_inv[rho,sigma]*(self.g[sigma, nu].diff(self.x[mu])+self.g[mu, sigma].diff(self.x[nu])-self.g[mu, nu].diff(self.x[sigma]))/2
        return ch
    
    def Lagrangian(self):
        lagrangian = 0

        for i in range(4):
            for j in range(4):
                lagrangian += self.g[i,j]*self.u[i]*self.u[j]/2
        
        return lagrangian
    
    def norm(self, x, vec):
        n = 0
        for i in range(4):
            for j in range(4):
                n += self.g_f(x)[i,j]*vec[i]*vec[j]
        return n