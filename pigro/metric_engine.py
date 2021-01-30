from sympy import *
import numpy as np
import json

#########################################################################################
#                               METRIC OBJECT                                           #
#                                                                                       #
#   The metric object contains the symbolic expresison of a generic metric tensor       #
#   and symbolically retrieves the inverse metric and all is needed to compute the      #
#   Christhoffel symbols.                                                               #
#   Its main contents are:                                                              #
#    -  g: the metrics tensor symbolic expression;                                      #
#    -  g_inv: the inverse metrics;                                                     #
#    -  x: an array of coordinates;                                                     #
#    -  u: an array of derivatives of x w.r.t. to the affine parameter;                 #
#                                                                                       #   
#   g and x are user inputs.                                                            #
#                                                                                       #
#########################################################################################

class metric():

    def __init__(self):
        self.initialized = 0
    
    def initialize_metric(self):
        self.name = input("Insert the name of the spacetime (e.g. 'Kerr metric'): ")

        print("Initialazing {}".format(self.name))
        
        print("Define coordinates symbols:")
        self.x = []
        self.x_str = []
        self.u = []
        for i in range(4):
            coordinate = input("Coordinate {}: ".format(i))
            setattr(self, coordinate, symbols(coordinate))
            self.x.append(self.__dict__[coordinate])
            self.x_str.append(coordinate)

            velocity = "u" + coordinate
            setattr(self, velocity, symbols(velocity))
            self.u.append(self.__dict__[velocity])
        
        print("Define metric tensor components:")
        self.g = zeros(4, 4)
        self.g_str = np.zeros([4,4], dtype = object)

        for i in range(4):
            for j in range(4):
                component = input("g[{},{}]: ".format(i, j))
                self.g[i,j] = parse_expr(component)
                self.g_str[i,j] = component
        
        self.g_f = lambdify([self.x], self.g, 'numpy')

        self.initialized = 1

        self.surfaces = {}
        self.constants = {}

        free_sym = list(self.g.free_symbols-set(self.x))

        if(len(free_sym)) > 0:
            for sym in free_sym:
                self.add_constant(str(sym))
            print("Symbols {} are not coordinate. They have been set as constants. Do you wish to set their value now?")
            if input("[y/n]: ") == "y":
                for sym in free_sym:
                    self.set_constant(sym, input("{} = ".format(str(sym))))
            else:
                pass
                
        print("Calculating inverse metric...")
        self.g_inv = self.g.inv()

        print("The metric_engine has been initialized.")

    def save_metric(self, filename):
        if self.initalized:
            f = open(filename, "w")
            
            output = {}
            
            output['name'] = self.name
            output['g'] = self.g_str.tolist()
            output['x'] = self.x_str
            if(self.transform_functions):
                output['transform'] = self.self.transform_functions_str

            json.dump(output, f)

            f.close()

        else:
            print("Inizialize (initialize_metric) or load (load_metric) a metric before saving.")
    
    def load_metric(self, filename, verbose = True, **params):

        f = open(filename, "r")

        load = json.load(f)

        self.name = load['name']

        if verbose:
            print("Loading {}".format(self.name))
        
        self.x = []
        self.x_str = []
        self.u = []

        for i in range(4):
            coordinate = load['x'][i]
            setattr(self, coordinate, symbols(coordinate))
            self.x.append(self.__dict__[coordinate])
            self.x_str.append(coordinate)

            velocity = "u" + coordinate
            setattr(self, velocity, symbols(velocity))
            self.u.append(self.__dict__[velocity])
        
        self.g = zeros(4, 4)
        self.g_str = np.zeros([4,4], dtype = object)

        for i in range(4):
            for j in range(4):
                component = load['g'][i][j]
                self.g[i,j] = parse_expr(component)
                self.g_str[i,j] = component

        self.initialized = 1

        self.horizons = {}
        self.constants = {}
        self.constants_of_motion = {}

        free_sym = list(self.g.free_symbols-set(self.x))

        if(len(free_sym)) > 0:
            for sym in free_sym:
                self.add_constant(str(sym))
                if str(sym) in params:
                    self.set_constant(sym, params.get(str(sym)))
                else:
                    self.set_constant(sym, input("{} = ".format(str(sym))))

        self.transform_functions = []

        if load['transform']:
            for i in range(3):
                transf = load['transform'][i]
                transform_function = parse_expr(transf)
                self.transform_functions.append(lambdify([self.x[1], self.x[2], self.x[3]], self.evaluate_constants(transform_function), 'numpy'))


        self.g_f = lambdify([self.x], self.evaluate_constants(self.g), 'numpy')

        if verbose:
            print("Calculating inverse metric tensor...")
            
        self.g_inv = self.g_val.inv()
        self.g_inv_s = self.g.inv()
        
        if verbose:
            print("The metric_engine has been initialized.")

    def add_constant(self, symbol):
        if self.initialized:
            self.constants[symbol] = {}
            self.constants[symbol]['symbol'] = symbol
            self.constants[symbol]['value'] = None
        else:
            print("Inizialize (initialize_metric) or load (load_metric) a metric before adding constants.")
    
    def set_constant(self, symbol, value):
        if self.initialized:
            self.constants[str(symbol)]['value'] = value
        else:
            print("Inizialize (initialize_metric) or load (load_metric) a metric before adding constants.")

    def evaluate_constants(self, expr):
        if any(x['value'] == None for x in self.constants.values()):
            print("You must set_constant for every constant value in your metric, before evaluating.")
        else:
            subs = []
            for c in self.constants.values():
                x = c['symbol']
                y = c['value']
                subs.append([x,y])
            return expr.subs(subs)
    
    def chr(self, mu, nu, rho):
        ch = 0
        for sigma in range(4):
            ch += self.g_inv_s[rho,sigma]*(self.g[sigma, nu].diff(self.x[mu])+self.g[mu, sigma].diff(self.x[nu])-self.g[mu, nu].diff(self.x[sigma]))/2
        return ch

    def set_coordinate_transformation(self):
        print("Define tranforamtion to pseudo-cartesian coordinates (useful for plotting):")
        self.transform_functions = []
        self.transform_functions_str = []
        coords = ["x", "y", "z"]
        for i in range(3):
            transf = input("{} = ".format(coords[i]))
            transform_function = parse_expr(transf)
            self.transform_functions_str.append(transf)
            self.transform_functions.append(lambdify([self.x[1], self.x[2], self.x[3]], self.evaluate_constants(transform_function), 'numpy'))

    def transform(self, X):
        return self.transform_functions[0](X[0], X[1], X[2]), self.transform_functions[1](X[0], X[1], X[2]), self.transform_functions[2](X[0], X[1], X[2])

    def set_constant_of_motion(self, name, expr):
        self.constants_of_motion[name] = lambdify([self.x, self.u], self.evaluate_constants(expr), 'numpy')