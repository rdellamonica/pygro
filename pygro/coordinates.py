import sympy as sp

class Coordinate:
    def __init__(self, name: str, parameter : str = "s"):
        self.name = name
        self.s = sp.symbols(parameter)
        self.x = sp.Function(self.name)(self.s)
        self.d = self.x.diff(self.s)
    
    def __repr__(self):
        return f"<Coordinate: {self.x}>"

class CoordinateSystem:
    def __init__(self, coordinates : list[str] | list[Coordinate], parameter: str = "s"):

        if all([isinstance(n, str) for n in coordinates]):
            self.names = coordinates
            self.x = [Coordinate(name, parameter) for name in coordinates]
            
        elif all([isinstance(n, Coordinate) for n in coordinates]):
            self.names = [str(coordinate) for coordinate in coordinates]
            self.x = coordinates
        
        else:
            raise TypeError("Coordinate systems only accepts lists of string or lists or Coordinates.")

        self.names_subs = {self.names[i]: self.x[i].x for i in range(4)}
            
    @property
    def d(self):
        return [x.d for x in self.x]

class CoordinateTransformation:
    def __init__(self, system_1: CoordinateSystem, system_2: CoordinateSystem, expressions: list[str]):
        
        self.expressions = [sp.parse_expr(expr).subs(system_1.names_subs) for expr in expressions]
        self.jacobian = sp.Matrix([[self.expressions[i].diff(system_1.x[j].x) for j in range(4)] for i in range(4)])
        
        if self.jacobian.det() == 0:
            raise ValueError("The transformation is non-invertible.")
        
cartesian_coordinates = CoordinateSystem(["t", "x", "y", "z"])