import numpy as np
from typing import Literal, Optional, Tuple, Callable, Optional

AVAILABLE_INTEGRATORS = Literal['rkf45', 'dp45', 'ck45', 'rkf78']

class IntegrationError(Exception):
    def __init__(self, message):
        self.message = message

class Integrator:
    """
        Base class for ODE integrator.
    """
    def __init__(self, function: Callable, stopping_criterion: Optional[Callable] = lambda x: True):
        
        if not callable(function):
            raise TypeError("'function' must be a callable object.")
        if not callable(stopping_criterion):
            raise TypeError("'function' must be a callable object.")
        
        self.f = function
        self.stopping_criterion = stopping_criterion
        
    def next_step(self, x: float, y: np.ndarray, h: float) -> Tuple[float, np.ndarray, float]:
        raise NotImplementedError("next_step is not implemented")
    
    def integrate(self, x_start: float, x_end: float, y_start: np.array, initial_step: float) -> Tuple[np.ndarray, np.ndarray, str]:
        x = [x_start]
        y = [y_start]

        h = initial_step
        
        exit = "failed"
        
        while abs(x[-1]) < abs(x_end) and self.stopping_criterion(*y[-1]):
            next = self.next_step(x[-1], y[-1], h, x_end)
            x.append(next[0])
            y.append(next[1])
            h = next[2]

            if not self.stopping_criterion(*y[-1]):
                exit = self.stopping_criterion.exit
            else:
                exit = "done"
        
        return np.array(x), np.array(y), exit

class ExplicitAdaptiveRungeKuttaIntegrator(Integrator):
    r"""
        Base class for explicit adaptive step-size integrators of the Runge-Kutta family.
        
        The local error is estimated as the difference between the two approximations at higher and lower order:

        .. math::

            E = \| y_{\text{high}} - y_{\text{low}} \|
        
        where :math:`\| \cdot \|` typically represents a norm, such as the maximum norm or the Euclidean norm (the one we use).
        
        The step size is adapted based on a defined error tolerance, which ensures the solution's accuracy, computed as:

        .. math::
            \text{tol} = \text{tol}_\text{abs} + \text{tol}_\text{rel} \cdot \| y \|
            :label: integration-tolerance
            
        The tolerance acts as a threshold for acceptable error in the solution. If :math:`E \leq \text{tol}`, the step is accepted; otherwise, it is rejected, and a smaller step size is used.
        
        The constructor for the :py:class:`.ExplicitAdaptiveRungeKuttaIntegrator` object accepts the following arguments that can be directly passed to the :py:meth:`~pygro.geodesic_engine.GeodesicEngine.integrate` method of the :py:class:`.GeodesicEngine` object upon integration:
        
        :param accuracy_goal: the exponent that defines the integration absolute tolerance ``atol = 10**(-accuracy_goal)``, :math:`\text{tol}_\text{abs}`, in formula :eq:`integration-tolerance`.
        :type accuracy_goal: int
        :param precision_goal: the exponent that defines the integration relative tolerance ``rtol = 10**(-precision_goal)``, :math:`\text{tol}_\text{rel}`,  in formula :eq:`integration-tolerance`.
        :type precision_goal: int
        :param hmax: maximum acceptable step size (default is ``hmax = 1e+16``).
        :type hmax: float
        :param hmin: minimum acceptable step size (default is ``hmin = 1e-16``).
        :type hmin: float
        :param max_iter: maxium namber of step-size adaptation attempts before rising an error. If this error is reached, try to use smaller ``accuracy_goal`` and/or ``precision_goal`` or to reduce the initial step size of the integration (``initial_step`` in the :py:meth:`~pygro.geodesic_engine.GeodesicEngine.integrate` method).
        :type max_iter: int
    """
    
    def __init__(self, function: Callable, stopping_criterion: Callable, order: int, stages: int, accuracy_goal: Optional[int] = 10, precision_goal: Optional[int] = 10, safety_factor: Optional[float] = 0.9, hmax: Optional[float] = 1e+16, hmin: Optional[float] = 1e-16, max_iter: int = 100):
        
        super().__init__(function, stopping_criterion)
        
        if not isinstance(order, int):
            raise TypeError("'order' must be integer.")
        if not isinstance(stages, int):
            raise TypeError("'stages' must be integer.")

        if not isinstance(accuracy_goal, int):
            raise TypeError("'accuracy_goal' must be integer.")
        if not isinstance(precision_goal, int):
            raise TypeError("'precision_goal' must be integer.")
        
        if not isinstance(safety_factor, (float, int)):
            raise TypeError("'safety_factor' must be a number.")
        if not isinstance(hmax, (float, int)):
            raise TypeError("'hmax' must be a number.")
        
        self.order = order
        self.stages = stages
    
        self.atol = 10**(-accuracy_goal)
        self.rtol = 10**(-precision_goal)
        self.sf = safety_factor
        self.hmax = hmax
        self.hmin = hmin
        self.min_factor = 0.2
        self.max_factor = 10
        self.max_iter = max_iter
    
    def next_step(self, x: float, y: np.ndarray, h: float, x_end: float) -> Tuple[float, np.ndarray, float]:
        k = np.zeros(self.stages, dtype = object)
        h1 = h
        
        j = 1
        
        while True:
            
            x1 = x + h1

            if np.sign(h1)*(x1 - x_end) > 0:
                x1 = x_end

            h1 = x1 - x
            
            for i in range(self.stages):
                k[i] = self.f(x+self.c[i]*h1, y + h1*np.dot(k, self.a[i]))
            
            y_lower = y + h1*np.dot(self.b[1], k)
            y_higher = y + h1*np.dot(self.b[0], k)
            
            v = np.linalg.norm(y_lower-y_higher)
            w = self.atol+self.rtol*np.linalg.norm(np.maximum(y_lower, y_higher))

            err = v/w
            
            if not err == 0: K = min(max(self.sf*err**(-1/self.order), self.min_factor), self.max_factor)

            if err > 1:
                # Reject step and try smaller step-size
                
                if j >= self.max_iter:
                    raise IntegrationError("Reached maximum number of step iterations. Try reducing the initial step size.")
                
                if h1 > 0:
                    h1 = max(min(abs(self.hmax), h1*K), abs(self.hmin))
                else:
                    h1 = -max(min(abs(self.hmax), abs(h1)*K), abs(self.hmin))
                continue
            
                j += 1
            
            else:
                # Accept step
                if err == 0:
                    if h1 > 0:
                        h2 = min(abs(self.hmax), 2*h1)
                    else:
                        h2 = -max(abs(self.hmax), 2*abs(h1))
                    break
                if h1 > 0:
                    h2 = max(min(abs(self.hmax), h1*K), abs(self.hmin))
                else:
                    h2 = -min(max(abs(self.hmax), abs(h1)*K), abs(self.hmin))
                break
            
        return x1, y_lower, h2
        
class DormandPrince45(ExplicitAdaptiveRungeKuttaIntegrator):
    def __init__(self, function: Callable, stopping_criterion: Callable, accuracy_goal: Optional[int] = 10, precision_goal: Optional[int] = 10, safety_factor: Optional[float] = 0.9, hmax: Optional[float] = 1e+16):

        self.a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])

        self.b = np.array([
            [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])

        self.c = np.array([
            0, 1/5, 3/10, 4/5, 8/9, 1, 1
        ])
        
        super().__init__(function, stopping_criterion, 5, len(self.b[0]), accuracy_goal, precision_goal, safety_factor, hmax)

class RungeKuttaFehlberg45(ExplicitAdaptiveRungeKuttaIntegrator):
    def __init__(self, function: Callable, stopping_criterion: Callable, accuracy_goal: Optional[int] = 10, precision_goal: Optional[int] = 10, safety_factor: Optional[float] = 0.9, hmax: Optional[float] = 1e+16):

        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0],
        ])

        self.b = np.array([
            [16/135,  0, 6656/12825, 28561/56430, -9/50, 2/55],
            [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        ])

        self.c = np.array([
            0, 1/4, 3/8, 12/13, 1, 1/2
        ])
        
        super().__init__(function, stopping_criterion, 5, len(self.b[0]), accuracy_goal, precision_goal, safety_factor, hmax)
    
    
class RungeKuttaFehlberg78(ExplicitAdaptiveRungeKuttaIntegrator):
    def __init__(self, function: Callable, stopping_criterion: Callable, accuracy_goal: Optional[int] = 10, precision_goal: Optional[int] = 10, safety_factor: Optional[float] = 0.9, hmax: Optional[float] = 1e+16):

        super().__init__(function, stopping_criterion, 7, 13, accuracy_goal, precision_goal, safety_factor, hmax)

        self.c = [0, 2/27,  1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]
        
        self.a = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2/27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/36, 1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/24, 0, 1/8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5/12, 0, -25/16, 25/16, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/20, 0, 0, 1/4, 1/5, 0, 0, 0, 0, 0, 0, 0, 0],
            [-25/108, 0, 0, 125/108, -65/27, 125/54, 0, 0, 0, 0, 0, 0, 0],
            [31/300, 0, 0, 0, 61/225, -2/9, 13/900, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3, 0, 0, 0, 0, 0],
            [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12, 0, 0, 0, 0],
            [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, 0, 0, 0],
            [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0, 0, 0],
            [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1, 0]
        ])

        self.b = np.array(
            [
                [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840],
                [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/840, 0, 0]
            ]
        )

class CashKarp45(Integrator):
    # TODO: improve CashKarp typing, notation and documentation
    def __init__(self, function: Callable, stopping_criterion: Callable, twiddle1 = 1.1, twiddle2 = 1.5, quit1 = 100, quit2 = 100, safety_factor = 0.9, accuracy_goal = 10, precision_goal = 10):
        
        super().__init__(function, stopping_criterion)

        self.twiddle1 = twiddle1
        self.twiddle2 = twiddle2
        self.quit1 = quit1
        self.quit2 = quit2
        self.SF = safety_factor
        self.tolerance = 1
        self.Atol = 10**(-accuracy_goal)
        self.Rtol = 10**(-precision_goal)
        
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0],
            [3/10, -9/10, 6/5, 0, 0, 0],
            [-11/54, 5/2, -70/27, 35/27, 0, 0],
            [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]
        ])

        self.b = np.array([
            [1, 0, 0, 0, 0, 0],
            [-3/2, 5/2, 0, 0, 0, 0],
            [19/54, 0, -10/27, 55/54, 0, 0],
            [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
            [37/378, 0, 250/621, 125/594, 0, 512/1771]
        ])

        self.c = np.array([
            0, 1/5, 3/10, 3/5, 1, 7/8
        ])

    def integrate(self, x_start, x_end, y_start, initial_step):

        x = [x_start]
        y = [y_start]

        h = initial_step
        twiddle1 = self.twiddle1
        twiddle2 = self.twiddle2
        quit1 = self.quit1
        quit2 = self.quit2

        while abs(x[-1]) <= abs(x_end) and self.stopping_criterion(*y[-1]):
            if self.verbose:
                print("{:.4f}".format(x[-1]), end= "\r")
            next = self.next_step(x[-1], y[-1], h, twiddle1, twiddle2, quit1, quit2)
            x.append(next[0])
            y.append(next[1])
            h = next[2]
            twiddle1 = next[3]
            twiddle2 = next[4]
            quit1 = next[5]
            quit2 = next[6]
        
        if not self.stopping_criterion(*y[-1]):
            exit = self.stopping_criterion.exit
        else:
            exit = "done"

        return np.array(x), np.array(y), exit
    
    def next_step(self, x, y, h, TWIDDLE1, TWIDDLE2, QUIT1, QUIT2):
        k = np.zeros(6, dtype = object)
        h1 = h
        while True:
            k[0] = h1*self.f(x, y)
            k[1] = h1*self.f(x + self.c[1]*h1, y + np.dot(k,self.a[1]))

            y1 = y + np.dot(self.b[0], k)
            y2 = y+ np.dot(self.b[1], k)
            
            E1 = self.E(y1, y2, 1)

            if E1 > TWIDDLE1*QUIT1:
                ESTTOL = E1/QUIT1
                h1 = max(1/5, self.SF/ESTTOL)*h1
                continue
                
            k[2] = h1*self.f(x + self.c[2]*h1, y+np.dot(k,self.a[2]))
            k[3] = h1*self.f(x + self.c[3]*h1, y+np.dot(k,self.a[3]))

            y3 = y + np.dot(self.b[2], k)
            E2 = self.E(y2, y3, 2)

            if E2 > TWIDDLE2*QUIT2:
                if E1 < 1:
                    if abs(1/10*max(k[0]+k[1])) < self.tolerance:
                        x = x + h1*self.c[1]
                        h1 = h1/5
                        return x, y2, h1, TWIDDLE1, TWIDDLE2, QUIT1, QUIT2
                    else:
                        h1 = h1/5
                        continue
                else:
                    ESTTOL = E2/QUIT2
                    h1 = max(1/5, self.SF/ESTTOL)*h1
                    continue
            
            k[4] = h1*self.f(x + self.c[4]*h1, y + np.dot(k,self.a[4]))
            k[5] = h1*self.f(x + self.c[5]*h1, y + np.dot(k,self.a[5]))

            y4 = y + np.dot(self.b[3], k)
            y5 = y + np.dot(self.b[4], k)

            E4 = self.E(y5, y4, 4)

            if E4 > 1:
                if E1/QUIT1 < TWIDDLE1:
                    TWIDDLE1 = max(1.1, E1/QUIT1)
                if E2/QUIT2 < TWIDDLE2:
                    TWIDDLE2 = max(1.1, E2/QUIT2)
                if E2 < 1:
                    if abs(1/10*max(k[0]-2*k[2]+k[3])) < self.tolerance:
                        x = x + h1*self.c[3]
                        h1 = h1*self.c[3]
                        return x, y3, h1, TWIDDLE1, TWIDDLE2, QUIT1, QUIT2
                else:
                    if E1 < 1:
                        if abs(1/10*max(k[0]+k[1])) < self.tolerance:
                            x = x + h1*self.c[1]
                            h1 = h1*self.c[1]
                            return x, y2, h1, TWIDDLE1, TWIDDLE2, QUIT1, QUIT2
                        else:
                            h1 = h1/5
                            continue
                    else:
                        ESTTOL = E4
                        h1 = max(1/5, self.SF/ESTTOL)*h1
                        continue
            else:
                x = x + h1*self.c[4]
                h1 = min(5, self.SF/E4)*h1

                Q1 = E1/E4
                if Q1 > QUIT1:
                    Q1 = min(Q1, 10*QUIT1)
                else:
                    Q1 = max(Q1, 2/3*QUIT1)
                QUIT1 = max(1, min(10000, Q1))

                Q2 = E2/E4
                if Q2 > QUIT2:
                    Qw = min(Q2, 10*QUIT2)
                else:
                    Q2 = max(Q2, 2/3*QUIT2)
                QUIT2 = max(1, min(10000, Q2))

                return x, y5, h1, TWIDDLE1, TWIDDLE2, QUIT1, QUIT2

    def E(self, y1, y2, i):
        return self.ERR(y1, y2, i)/self.tolerance**(1/(1+i))

    def ERR(self, y1, y2, i):
        if hasattr(y1, "__len__"):
            v = y1-y2
            w = v/(self.Atol+self.Rtol*y1)
            return np.linalg.norm(w)**(1/(1+i))
        else:
            v = y1-y2
            return abs(v/(self.Atol+self.Rtol*y1))**(1/(1+i))

def get_integrator(integrator: AVAILABLE_INTEGRATORS, *args, **kwargs) -> Integrator:
    if integrator == "rkf45":
        return RungeKuttaFehlberg45(*args, **kwargs)
    if integrator == "dp45":
        return DormandPrince45(*args, **kwargs)
    if integrator == "ck45":
        return CashKarp45(*args, **kwargs)
    if integrator == "rkf78":
        return RungeKuttaFehlberg78(*args, **kwargs)
    