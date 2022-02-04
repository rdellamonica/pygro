import sympy as sp
import numpy as np
import json
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.core.function import AppliedUndef

#########################################################################################
#                               METRIC OBJECT                                           #
#                                                                                       #
#   The metric object contains the symbolic expresison of a generic metric tensor       #
#   and symbolically retrieves the inverse metric and all is needed to compute the      #
#   Christoffel symbols.                                                                #
#   Its main contents are:                                                              #
#    -  g: the metrics tensor symbolic expression;                                      #
#    -  g_inv: the inverse metrics;                                                     #
#    -  x: an array of coordinates;                                                     #
#    -  u: an array of derivatives of x w.r.t. to the affine parameter;                 #
#                                                                                       #   
#   g and x are user inputs.                                                            #
#                                                                                       #
#########################################################################################

def parse_expr(expr):
    return sp.parse_expr(expr)

class Metric():
    def __init__(self, **kwargs):
        self.initialized = 0
        self.initialized_metric = 0
        self.geodesic_engine_linked = False

        if len(kwargs) > 0:
            if "load" in kwargs:
                self.load_metric(kwargs["load"])
            self.initialize_metric(**kwargs)
    
    def initialize_metric(self, **kwargs):

        self.x = []
        self.dx = []
        self.x_str = []
        self.u = []

        interactive_metric_insertion = len(kwargs) == 0        

        if not interactive_metric_insertion:

            minimal_kwargs = ["name", "coordinates", "line_element"]

            for kwarg in minimal_kwargs:
                if not kwarg in kwargs:
                    raise ValueError('initialize_metric in non-interactive mode should have "name", "coordinates", "line_element" as minimal arguments')

            self.name = kwargs['name']

            coordinates_input = kwargs["coordinates"]
            
            if len(coordinates_input) != 4:
                raise ValueError('coordinates should be a 4-dimensional list of strings')
            
            for coordinate in coordinates_input:
                setattr(self, coordinate, sp.symbols(coordinate))
                self.x.append(self.__dict__[coordinate])
                self.x_str.append(coordinate)

                velocity = "u_" + coordinate
                setattr(self, velocity, sp.symbols(velocity))
                self.u.append(self.__dict__[velocity])

                differential = "d" + coordinate
                setattr(self, differential, sp.symbols(differential))
                self.dx.append(self.__dict__[differential])
            
            self.g = sp.zeros(4, 4)
            self.g_str = np.zeros([4,4], dtype = object)

            try:
                ds2_str = kwargs["line_element"]
                ds2_sym = sp.expand(sp.parse_expr(ds2_str))
            except:
                raise ValueError("Please insert a valid expression for the line element.")
            else:
                self.ds2 = ds2_sym
                for i, dx1 in enumerate(self.dx):
                    for j, dx2 in enumerate(self.dx):
                        self.g[i,j] = self.ds2.coeff(dx1*dx2,1)
                        self.g_str[i,j] = str(self.g[i,j])

        else:
            self.name = input("Insert the name of the spacetime (e.g. 'Kerr metric'): ")

            print("Initialazing {}".format(self.name))
            
            print("Define coordinates symbols:")
            
            for i in range(4):
                coordinate = input("Coordinate {}: ".format(i))
                setattr(self, coordinate, sp.symbols(coordinate))
                self.x.append(self.__dict__[coordinate])
                self.x_str.append(coordinate)

                velocity = "u_" + coordinate
                setattr(self, velocity, sp.symbols(velocity))
                self.u.append(self.__dict__[velocity])

                differential = "d" + coordinate
                setattr(self, differential, sp.symbols(differential))
                self.dx.append(self.__dict__[differential])
            

            case = input("From? [tensor/line element]: ")

            if case == "tensor":
                print("Define metric tensor components:")
                self.g = sp.zeros(4, 4)
                self.g_str = np.zeros([4,4], dtype = object)

                for i in range(4):
                    for j in range(4):
                        while True:
                            try:
                                component = input("g[{},{}]: ".format(i, j))
                                component_symb = sp.parse_expr(component)
                            except:
                                print("Please insert a valid expression for the component.")
                                continue
                            else:
                                self.g[i,j] = component_symb
                                self.g_str[i,j] = component
                                break
                        
            elif case == "line element":
                self.g = sp.zeros(4, 4)
                self.g_str = np.zeros([4,4], dtype = object)
                while True:
                    try:
                        ds2_str = input("ds^2 = ")
                        ds2_sym = sp.expand(sp.parse_expr(ds2_str))
                    except:
                        print("Please insert a valid expression for the line element.")
                        continue
                    else:
                        self.ds2 = ds2_sym
                        for i, dx1 in enumerate(self.dx):
                            for j, dx2 in enumerate(self.dx):
                                self.g[i,j] = self.ds2.coeff(dx1*dx2,1)
                                self.g_str[i,j] = str(self.g[i,j])
                        break
            else:
                raise("Only 'tensor' or 'line element' are accepted method for parsing the metric.")
            
        print("Calculating inverse metric...")
        self.g_inv = self.g.inv()
        self.g_inv_str = np.zeros([4,4], dtype = object)
        
        for i in range(4):
            for j in range(4):
                self.g_inv_str[i,j] = str(self.g_inv[i,j])
        
        print("Calculating symbolic equations of motion:")
        self.eq_u = []
        self.eq_u_str = np.zeros(4, dtype = object)

        self.eq_x = []
        self.eq_x_str = np.zeros(4, dtype = object)

        for rho in range(4):
            print("- {}/4".format(rho+1))
            eq = 0
            for mu in range(4):
                for nu in range(4):
                    eq += -self.Christoffel(mu, nu, rho)*self.u[mu]*self.u[nu]
            self.eq_u.append(eq)
            self.eq_u_str[rho] = str(eq)

            self.eq_x.append(self.u[rho])
            self.eq_x_str[rho] = str(self.u[rho])


        print("Adding to class a method to get initial u_0...")

        eq = 0

        for mu in range(4):
            for nu in range(4):
                eq += self.g[mu, nu]*self.u[mu]*self.u[nu]

        self.u0_s_null = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]

        eq += 1

        self.u0_s_timelike = sp.solve(eq, self.u[0], simplify=False, rational=False)[0]

        self.initialized = 1
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
                    value = float(input("Insert value for {}: ".format(str(sym))))
                    self.set_constant(**{str(sym): value})

        if len(free_func) > 0:
            for func in free_func:
                self.add_parameter(str(func))
                
                if interactive_metric_insertion:
                    kind = input("Define kind for function {} [expr/py]: ".format(str(func)))
                else:
                    if not str(func.func) in kwargs:
                        raise ValueError("Auxiliary functions shoudld be passed as arguments, either as strings (expression mode) or as python methods (functional mode).")
                    
                    if isinstance(kwargs[str(func.func)], str):
                        kind = "expr"
                    elif callable(kwargs[str(func.func)]):
                        kind = "py"

                if kind == "expr":
                    self.parameters[str(func)]['kind'] = "expression"
                    if interactive_metric_insertion:
                        expr_str = input("{} = ".format(str(func)))
                    else:
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

                    if interactive_metric_insertion:
                        print(f"Remember to set the Python function for {str(func)} and its derivatives with")
                    
                        derlist = ""
                        for arg in list(func.args):
                            derlist += f", d{str(func.func)}d{str(arg)} = func"
                        
                        print(f"set_function_to_parameter({str(func)}, f = func{derlist})")

                    else:
                        args = list(func.args)
                        derlist = [f"d{str(func.func)}d{str(arg)}" for arg in args]

                        arg_derlist = {}

                        for der in derlist:
                            if not der in kwargs:
                                raise ValueError(f"In functional mode pass as arguments functional derivatives  of auxiliary functions. In this case '{der}' is missing.")
                            else:
                                arg_derlist[der] = kwargs[der]

                        self.set_function_to_parameter(str(func), kwargs[str(func.func)], **arg_derlist)

            if interactive_metric_insertion:
                while True:
                    case = input("Want to insert transform functions to pseudo-cartesian coordiantes? [y/n] ")

                    if case == "y":
                        for x in ["t", "x", "y", "z"]:
                            while True:
                                try:
                                    x_inpt = input(f"{x} = ")
                                    x_symb = sp.parse_expr(x_inpt)
                                except KeyboardInterrupt:
                                    raise SystemExit
                                except:
                                    print("Insert a valid expression.")
                                    continue
                                else:
                                    self.transform_s.append(x_symb)
                                    self.transform_functions_str.append(x_inpt)
                                    self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(x_symb), 'numpy'))
                                    break
                        break
                                    
                    elif case == "n":
                        break
                    else:
                        print("Not a valid input.")
                        continue
            else:
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
                    

        print("The metric_engine has been initialized.")

    def save_metric(self, filename):
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
            print("Initialize (initialize_metric) or load (load_metric) a metric before saving.")
    
    def load_metric(self, filename, verbose = True, **params):

        f = open(filename, "r")

        load = json.load(f)

        self.load_metric_from_json(load, verbose, **params)

        
    def load_metric_from_json(self, metric_json, verbose = True, **params):

        self.json = metric_json
        load = metric_json

        self.name = load['name']

        if not "expressions" in load:
            load["expressions"] = []
        
        if not "functions" in load:
            load["functions"] = []

        if verbose:
            print("Loading {}".format(self.name))
        
        self.x = []
        self.dx = []
        self.x_str = []
        self.u = []

        for i in range(4):

            coordinate = load['x'][i]
            setattr(self, coordinate, sp.symbols(coordinate))
            self.x.append(self.__dict__[coordinate])
            self.x_str.append(coordinate)

            velocity = "u_" + coordinate
            setattr(self, velocity, sp.symbols(velocity))
            self.u.append(self.__dict__[velocity])

            differential = "d" + coordinate
            setattr(self, differential, sp.symbols(differential))
            self.dx.append(self.__dict__[differential])
        
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
        
        self.initialized = 1

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
                print(f"Remember to set the Python function for {str(func)}.")
            
            for arg in args:
                dername = f"d{func}d{str(arg)}"

                if dername in params:
                    self.parameters[function['name']][dername] = params[dername]
                else:
                    print(f"Remember to set the Python function for {dername}.")

        self.initialized_metric = 1
        
        if "transform" in load:
            for i in range(4):
                transf = load['transform'][i]
                transform_function = sp.parse_expr(transf)
                self.transform_s.append(transform_function)
                self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(transform_function), 'numpy'))

        if verbose:
            print("The metric_engine has been initialized.")

    def add_parameter(self, symbol):
        if self.initialized:
            self.parameters[symbol] = {}
            self.parameters[symbol]['symbol'] = symbol
            self.parameters[symbol]['symbolic'] = sp.parse_expr(symbol)
            self.parameters[symbol]['value'] = None
            self.parameters[symbol]['kind'] = None
        else:
            print("Initialize (initialize_metric) or load (load_metric) a metric before adding parameters.")
    
    def set_constant(self, **params):
        if self.initialized:
            for param in params:
                try:
                    self.parameters[str(param)]['value'] = params[param]
                    self.parameters[str(param)]['kind'] = "constant"
                except:
                    print(f"No parameter named '{str(param)}' in the metric_engine.")
                    break
        else:
            print("Initialize (initialize_metric) or load (load_metric) a metric before adding parameters.")
    
    def set_expression_to_parameter(self, param, expr_str):
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
                raise TypeError(f"No parameter named '{str(param)}' in the metric_engine.")
        else:
            print("Initialize (initialize_metric) or load (load_metric) a metric before adding parameters.")


    def set_function_to_parameter(self, param, function, **derivatives):
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
            print("Initialize (initialize_metric) or load (load_metric) a metric before adding parameters.")

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
            print("You must set_constant for every constant value in your metric, before evaluating.")
        else:
            subs = []
            const = self.get_parameters_constants()
            for c in const:
                x = const[c]['symbol']
                y = const[c]['value']
                subs.append([x,y])
            return self.subs_functions(expr).subs(subs)

    def set_coordinate_transformation(self):
            self.transform_s = []
            self.transform_functions = []
            for x in ["t", "x", "y", "z"]:
                while True:
                    try:
                        x_inpt = input(f"{x} = ")
                        x_symb = sp.parse_expr(x_inpt)
                    except KeyboardInterrupt:
                        raise SystemExit
                    except:
                        print("Insert a valid expression.")
                        continue
                    else:
                        self.transform_s.append(x_symb)
                        self.transform_functions.append(sp.lambdify([self.x], self.evaluate_parameters(x_symb), 'numpy'))
                        break
    
    def parse_expr(self, expr):
        return sp.parse_expr(expr)

    def transform(self, X):
        if self.transform_functions:
            return self.transform_functions[0](X), self.transform_functions[1](X), self.transform_functions[2](X), self.transform_functions[3](X)
        else:
            raise TypeError("Coordinate transformations not set. Consider using .set_coordinate_transformation method in Metric class.")

    '''def norm4(self, x, v):
        norm = 0
        
        for mu in range(4):
            for nu in range(4):
                norm += self.g_f(x)[mu, nu]*v[mu]*v[nu]
        
        return norm'''
    
    def g_f(self, x):
        return lambdify([*self.x, *self.get_parameters_symb()], self.subs_functions(self.g))(*x, *self.get_parameters_val())

    def Christoffel(self, mu, nu, rho):
        ch = 0
        for sigma in range(4):
            ch += self.g_inv[rho,sigma]*(self.g[sigma, nu].diff(self.x[mu])+self.g[mu, sigma].diff(self.x[nu])-self.g[mu, nu].diff(self.x[sigma]))/2
        return ch