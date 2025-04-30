import numpy as np
from collections.abc import Iterable

class Interpolator:
    """
    Base class for interpolation methods.
    
    :param x: Array of x values.
    :type x: np.ndarray
    :param y: Array of y values.
    :type y: np.ndarray
    :param k: List of derivative arrays.
    :type k: list[np.ndarray]
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, k: list[np.ndarray]):
        self.x = x
        self.y = y
        self.k = k
        
    def _compute_interpolation(self, x: float) -> np.ndarray:
        """
        Compute the interpolated value at a given x.
        
        :param x: Point at which to evaluate the interpolation.
        :type x: float
        :raises NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError
    
    def _interpolate(self, x: float) -> np.ndarray:
        """
        Interpolate a value within the defined range.
        
        :param x: Point at which to interpolate.
        :type x: float
        :raises ValueError: If x is outside the integration range.
        :return: Interpolated value.
        :rtype: np.ndarray
        """
        if not min(self.x) <= x <= max(self.x):
            raise ValueError(f"Provided value ({x}) is outside the integration range [{min(self.x)}, {max(self.x)}].")
        
        if x == self.x[0]:
            return self.y[0]
        
        if x == self.x[-1]:
            return self.y[-1]
        
        return self._compute_interpolation(x)
    
    def __call__(self, x: float) -> float | np.ndarray:
        """
        Evaluate the interpolator at a given x.
        
        :param x: Point or array of points at which to evaluate.
        :type x: float or Iterable
        :return: Interpolated value(s).
        :rtype: float | np.ndarray
        """
        if isinstance(x, Iterable):
            return np.array([self._interpolate(x_i) for x_i in x])
        return np.array([self._interpolate(x)])
    
class LinearInterpolator(Interpolator):
    """
    Placeholder for a linear interpolation scheme. Currently not implemented.
    """
    pass

class HermiteCubicInterpolator(Interpolator):
    """
    Implements Hermite cubic interpolation.
    
    This method ensures smooth interpolation by using derivative information.
    """
    def _compute_interpolation(self, x):
        idx = np.searchsorted(self.x, x)-1
        
        x_0 = self.x[idx]
        x_1 = self.x[idx+1]
        y_0 = self.y[idx]
        y_1 = self.y[idx+1]
        h = x_1-x_0
        k_0 = self.k[idx]
        k_1 = self.k[idx+1]
        
        t = (x-x_0)/h
        
        return (1 - t) * y_0 + t * y_1 + t * (t - 1) * ((1 - 2 * t) * (y_1 - y_0) + (t - 1) * h * k_0[0] + t * h * k_1[0])
    
class DP54DenseOutput(Interpolator):
    """
    Dense output interpolator for the Dormand-Prince 5(4) method.
    """
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        
    def _compute_interpolation(self, x):
        idx = np.searchsorted(self.x, x)-1
        
        x_0 = self.x[idx]
        x_1 = self.x[idx+1]
        y_0 = self.y[idx]
        h = x_1-x_0 
        k = self.k[idx]
        
        t = (x-x_0)/h
        
        b_0 = t**2 * (3 - 2*t) * self.b[0] + t * (t - 1)**2 - t**2 * (t - 1)**2 * 5 * (2558722523 - 31403016 * t) / 11282082432
        b_1 = 0
        b_2 = t**2 * (3 - 2*t) * self.b[2] + t**2 * (t - 1)**2 * 100 * (882725551 - 15701508 * t) / 32700410799
        b_3 = t**2 * (3 - 2*t) * self.b[3] - t**2 * (t - 1)**2 * 25 * (443332067 - 31403016 * t) / 1880347072
        b_4 = t**2 * (3 - 2*t) * self.b[4] + t**2 * (t - 1)**2 * 32805 * (23143187 - 3489224 * t) / 199316789632
        b_5 = t**2 * (3 - 2*t) * self.b[5] - t**2 * (t - 1)**2 * 55 * (29972135 - 7076736 * t) / 822651844
        b_6 = t**2 * (t - 1) + t**2 * (t - 1)**2 * 10 * (7414447 - 829305 * t)/29380423
        
        b = np.array([b_0, b_1, b_2, b_3, b_4, b_5, b_6])
        
        return y_0 + h*np.dot(b, k)