import numpy as np
import time
import sympy as sp

################################################################
#   The geodesic object needs as arguments:                    #
#    -  a type string with "null" or "time-like"               #
#       for different kinds of geodesics.                      #
#    -  the engine object                                      #
################################################################


class Geodesic():
    def __init__(self, type, engine, verbose = True):
        if not type in ['null', 'time-like']:
            raise TypeError("Geodesic must be either 'null' or 'time-like'.")
        self.initial_x = []
        self.initial_u = []
        self.tau = []
        self.x = []
        self.u = []
        self.type = type
        self.engine = engine
        self.metric = engine.metric
        self.verbose = verbose
    
    def get_initial_u0(self, u1, u2, u3):
        if self.type == "null":
            u0_sol = self.engine.u0_f_null(self.initial_x, u1, u2, u3)
        elif self.type == "time-like":
            u0_sol = self.engine.u0_f_timelike(self.initial_x, u1, u2, u3)
        return u0_sol
    
    def get_initial_u1(self, u0, u2, u3):
        if self.type == "null":
            u1_sol = self.engine.u1_f_null(self.initial_x, u0, u2, u3)
        elif self.type == "time-like":
            u1_sol = self.engine.u1_f_timelike(self.initial_x, u0, u2, u3)
        return u1_sol

    def set_starting_point(self, x0, x1, x2, x3):
        if self.verbose:
            print("Setting starting point")
        self.initial_x = np.array([x0, x1, x2, x3])

    def set_starting_velocity_direction(self, theta, phi, v = 1, angles="rad"):
        self.u = []
        if(len(self.initial_x)>0):
            if self.verbose:
                print("Setting pointing direction")
            if angles == "rad":
                u1 = -v*np.cos(theta)*np.cos(phi)/np.sqrt(self.metric.g_f(self.initial_x)[1,1])
                u2 = -v*np.sin(theta)/np.sqrt(self.metric.g_f(self.initial_x)[2,2])
                u3 = -v*np.cos(theta)*np.sin(phi)/np.sqrt(self.metric.g_f(self.initial_x)[3,3])
                u0 = self.get_initial_u0(u1, u2, u3)
                self.initial_u = np.array([u0, u1, u2, u3])
            elif angles == "deg":
                theta = np.deg2rad(theta)
                phi = np.deg2rad(phi)
                u1 = -v*np.cos(theta)*np.cos(phi)/np.sqrt(self.metric.g_f(self.initial_x)[1,1])
                u2 = -v*np.sin(theta)/np.sqrt(self.metric.g_f(self.initial_x)[2,2])
                u3 = -v*np.cos(theta)*np.sin(phi)/np.sqrt(self.metric.g_f(self.initial_x)[3,3])
                u0 = self.get_initial_u0(u1, u2, u3)
                self.initial_u = np.array([u0, u1, u2, u3])
            else:
                print("angle can only be rad or deg")
        else:
            print("You must set_starting_point before initializing the 4-velocity.")
    
    def set_starting_4velocity(self, u1, u2, u3):
        self.u = []
        if(len(self.initial_x)>0):
            if self.verbose:
                print("Setting pointing direction")
            u0 = self.get_initial_u0(self.initial_x, u1, u2, u3)
            self.initial_u = np.array([u0, u1, u2, u3])
        else:
            print("You must set_starting_point before initializing the 4-velocity.")

    def lagrangian(self):
        self.L = 0
        for mu in range(4):
            for nu in range(4):
                self.L += self.g[mu,nu]*self.u[mu]*self.u[nu]/2

        L_f = sp.lambdify([self.x, self.u], self.L, 'numpy')

        def f(self, x, u):
            return L_f(x, u)

        self.__class__.L_f = f