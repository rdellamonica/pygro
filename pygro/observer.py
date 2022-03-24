import sympy as sp
import numpy as np

class Observer:
    
    def __init__(self, metric, x, frame, fromframe = False):
        
        self.metric = metric
        self.x = x
        
        if len(frame) != 4:
                raise ValueError('coframe should be a 4-dimensional list of strings')
        
        if fromframe:
            self.get_frame(frame)
        else:
            self.get_coframe(frame)
    
    def evaluate_parameters(self, expr):
        return self.metric.evaluate_parameters(expr)
    
    def evaluate_expression(self, expr):
        return sp.lambdify([*self.metric.x, *self.metric.get_parameters_symb()], self.metric.subs_functions(expr))(*self.x, *self.metric.get_parameters_val())
    
    def get_coframe(self, coframe):
        
        coframe_symb = []
        self.coframe_matrix = sp.zeros(4, 4)
    
        try:
            for coframe_i in coframe:
                coframe_symb.append(sp.expand(sp.parse_expr(coframe_i)))
        except:
            raise ValueError("Please insert a valid expression for the coframes components.")
        else:
            for i, coframe_i in enumerate(coframe_symb):
                for j, dx in enumerate(self.metric.dx):
                        self.coframe_matrix[i, j] = coframe_i.coeff(dx,1)
            
            self.frame_matrix = self.coframe_matrix.inv()
    
    def get_frame(self, frame):
        
        frame_symb = []
        self.frame_matrix = sp.zeros(4, 4)
    
        try:
            for frame_i in frame:
                frame_symb.append(sp.expand(sp.parse_expr(frame_i)))
        except:
            raise ValueError("Please insert a valid expression for the frames components.")
        else:
            for i, frame_i in enumerate(frame_symb):
                for j, x in enumerate(self.metric.x):
                        self.frame_matrix[i, j] = frame_i.coeff(sp.symbols("e"+str(x)),1)
            
            self.coframe_matrix = self.frame_matrix.inv()
            
    def convert_3vector(self, vector, kind = "time-like"):
        
        u_123 = self.evaluate_expression(self.frame_matrix[1:, 1:])@np.array(vector)
        
        if kind == "time-like":
            u0 = abs(sp.lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(self.metric.u0_s_timelike))(*self.x, *u_123, *self.metric.get_parameters_val()))
        elif kind == "null":
            u0 = abs(sp.lambdify([*self.metric.x, self.metric.u[1], self.metric.u[2], self.metric.u[3], *self.metric.get_parameters_symb()], self.metric.subs_functions(self.metric.u0_s_null))(*self.x, *u_123, *self.metric.get_parameters_val()))
        else:
            raise ValueError("Only 'time-like' and 'null' can be accepted as 'kind'.")
        
        return np.append(u0, u_123)
    
    def from_frame_vector(self, theta_obs, phi_obs, kind = "time-like", angles = "rad ", **kwargs):
        
        if kind == "time-like":
            if "v" in kwargs:
                v = kwargs["v"]
            else:
                raise ValueError("You must set the modulus 'v' for a time-like vector")
        else:
            v = 1
        
        if angles == "deg":
            theta_obs = np.deg2rad(theta_obs)
            phi_obs = np.deg2rad(phi_obs)
        elif angles != "rad":
            raise ValueError("Angles can only be 'deg' or 'rad'.")

        x_obs = v*np.cos(theta_obs)*np.cos(phi_obs)
        y_obs = v*np.sin(theta_obs)
        z_obs = v*np.cos(theta_obs)*np.sin(phi_obs)
        
        return x_obs, y_obs, z_obs
    
    def from_f1(self, theta_obs, phi_obs, kind = "time-like", **kwargs):
        x, y, z = self.from_frame_vector(theta_obs, phi_obs, kind, **kwargs)
        return self.convert_3vector([x, y, z], kind)
    
    def from_f2(self, theta_obs, phi_obs, kind = "time-like", **kwargs):
        x, y, z = self.from_frame_vector(theta_obs, phi_obs, kind, **kwargs)
        return self.convert_3vector([z, x, y], kind)
    
    def from_f3(self, theta_obs, phi_obs, kind = "time-like", **kwargs):
        x, y, z = self.from_frame_vector(theta_obs, phi_obs, kind, **kwargs)
        return self.convert_3vector([z, y, x], kind)