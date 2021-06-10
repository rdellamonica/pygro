import numpy as np
import scipy.interpolate as sp_int

def get_integrator(integrator, *args, **kwargs):
    if integrator == "rkf45":
        return RungeKuttaFehlberg45(*args, **kwargs)
    if integrator == "dp45":
        return DormandPrince45(*args, **kwargs)
    if integrator == "ck45":
        return CashKarp45(*args, **kwargs)
    if integrator == "rkf78":
        return RungeKuttaFehlberg78(*args, **kwargs)
        

class CashKarp45():
    def __init__(self, f, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 0, twiddle1 = 1.1, twiddle2 = 1.5, quit1 = 100, quit2 = 100, SF = 0.9, interpolate = False, stopping_criterion = "none", verbose = False):

        self.twiddle1 = twiddle1
        self.twiddle2 = twiddle2
        self.quit1 = quit1
        self.quit2 = quit2
        self.SF = SF
        self.f = f
        self.tolerance = 1
        self.Atol = 10**(-AccuracyGoal)
        self.Rtol = 10**(-PrecisionGoal)
        self.initial_step = initial_step
        self.interpolate = interpolate
        self.verbose = verbose

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

        if stopping_criterion == "none":
            self.stopping_criterion = lambda geo: True
        else:
            self.stopping_criterion = stopping_criterion

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
        
        if not self.interpolate:
            return np.array(x), np.array(y)
        else:
            return sp_int.interp1d(np.array(x), np.array(y), kind = 'cubic')

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

class RungeKuttaFehlberg78():
    def __init__(self, f, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 0, SF = 0.9, interpolate = False, stopping_criterion = "none", verbose = False):

        self.f = f
        self.tolerance = 1
        self.Atol = 10**(-AccuracyGoal)
        self.Rtol = 10**(-PrecisionGoal)
        self.initial_step = initial_step
        self.interpolate = interpolate
        self.verbose = verbose
        self.SF = SF

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
                [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/180, 0, 0],
                [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/180, 41/180]
            ]
        )

        if stopping_criterion == "none":
            self.stopping_criterion = lambda geo: True
        else:
            self.stopping_criterion = stopping_criterion

    def integrate(self, x_start, x_end, y_start, initial_step):

        x = [x_start]
        y = [y_start]

        h = initial_step

        while abs(x[-1]) <= abs(x_end) and self.stopping_criterion(*y[-1]):
            if self.verbose:
                print("{:.4f}".format(x[-1]), end= "\r")
            next = self.next_step(x[-1], y[-1], h)
            x.append(next[0])
            y.append(next[1])
            h = next[2]
        
        if not self.interpolate:
            return np.array(x), np.array(y)
        else:
            return sp_int.interp1d(np.array(x), np.array(y), kind = 'cubic')

    def next_step(self, x, y, h):
        k = np.zeros(13, dtype = object)
        h1 = h
        while True:
            
            for i in range(13):
                k[i] = h1*self.f(x+self.c[i]*h1, y + np.dot(k, self.a[i]))
            
            y7 = y + np.dot(self.b[0], k)
            y8 = y + np.dot(self.b[1], k)
            
            v = y8-y7
            w = abs(v)/(self.Atol+self.Rtol*abs(y7))

            err = max(w)
            print(err)
            if err > 1:
                h1 *= self.SF*err**(-1/7)
                continue
            else:
                h1 *= self.SF*err**(-1/8)
                break
        x1 = x + h1
        return x1, y8, h1

class RungeKuttaFehlberg45():
    def __init__(self, f, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 0, SF = 0.9, interpolate = False, stopping_criterion = "none", verbose = False, hmax = 1e+16):

        self.f = f
        self.tolerance = 1
        self.Atol = 10**(-AccuracyGoal)
        self.Rtol = 10**(-PrecisionGoal)
        self.initial_step = initial_step
        self.interpolate = interpolate
        self.verbose = verbose
        self.SF = SF
        self.hmax = hmax

        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ])

        self.b = np.array([
            [16/135,  0, 6656/12825, 28561/56430, -9/50, 2/55],
            [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
        ])

        self.c = np.array([
            0, 1/4, 3/8, 12/13, 1, 1/2
        ])

        if stopping_criterion == "none":
            self.stopping_criterion = lambda geo: True
        else:
            self.stopping_criterion = stopping_criterion

    def integrate(self, x_start, x_end, y_start, initial_step):

        x = [x_start]
        y = [y_start]

        h = initial_step

        while abs(x[-1]) <= abs(x_end) and self.stopping_criterion(*y[-1]):
            if self.verbose:
                print("{:.4f}".format(x[-1]), end= "\r")
            next = self.next_step(x[-1], y[-1], h)
            x.append(next[0])
            y.append(next[1])
            h = next[2]
        
        if not self.interpolate:
            return np.array(x), np.array(y)
        else:
            return sp_int.interp1d(np.array(x), np.array(y), kind = 'cubic')

    def next_step(self, x, y, h):
        k = np.zeros(6, dtype = object)
        h1 = h
        while True:
            
            for i in range(6):
                k[i] = h1*self.f(x+self.c[i]*h1, y + np.dot(k, self.a[i]))
            
            y4 = y + np.dot(self.b[1], k)
            y5 = y + np.dot(self.b[0], k)
            
            v = y4-y5
            w = abs(v)/(self.Atol+self.Rtol*abs(y4))

            err = max(w)
            if err > 1:
                h1 *= self.SF*err**(-0.25)
                continue
            else:
                if err == 0:
                    h2 = min(self.hmax, 2*h1)
                    break
                h2 = min(self.hmax, h1*self.SF*err**(-0.2))
                break
        x1 = x + h1
        return x1, y5, h2


class DormandPrince45():
    def __init__(self, f, initial_step = 1, AccuracyGoal = 10, PrecisionGoal = 10, SF = 0.9, interpolate = False, stopping_criterion = "none", verbose = False, hmax = 1e+16):

        self.f = f
        self.tolerance = 1
        self.Atol = 10**(-AccuracyGoal)
        self.Rtol = 10**(-PrecisionGoal)
        self.initial_step = initial_step
        self.interpolate = interpolate
        self.verbose = verbose
        self.SF = SF
        self.hmax = hmax

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
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
            [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
        ])

        self.c = np.array([
            0, 1/5, 3/10, 4/5, 8/9, 1, 1
        ])

        if stopping_criterion == "none":
            self.stopping_criterion = lambda geo: True
        else:
            self.stopping_criterion = stopping_criterion

    def integrate(self, x_start, x_end, y_start, initial_step):

        x = [x_start]
        y = [y_start]

        h = initial_step

        while abs(x[-1]) <= abs(x_end) and self.stopping_criterion(*y[-1]):
            if self.verbose:
                print("{:.4f}".format(x[-1]), end= "\r")
            next = self.next_step(x[-1], y[-1], h)
            x.append(next[0])
            y.append(next[1])
            h = next[2]
        
        if not self.interpolate:
            return np.array(x), np.array(y)
        else:
            return sp_int.interp1d(np.array(x), np.array(y), kind = 'cubic')

    def next_step(self, x, y, h):
        k = np.zeros(7, dtype = object)
        h1 = h
        while True:
            
            for i in range(7):
                k[i] = self.f(x+self.c[i]*h1, y + h1*np.dot(k, self.a[i]))
            
            y4 = y + h1*np.dot(self.b[1], k)
            y5 = y + h1*np.dot(self.b[0], k)
            
            v = y4-y5
            w = abs(v)/(self.Atol+self.Rtol*abs(np.maximum(y4,y5)))

            err = np.linalg.norm(w) / w.size ** 0.5

            if err > 1:
                h1 *= self.SF*err**(-0.2)
                continue
            else:
                if err == 0:
                    if self.hmax > 0:
                        h2 = min(self.hmax, 2*h1)
                    else:
                        h2 = max(self.hmax, 2*h1)
                    break
                if self.hmax > 0:
                    h2 = min(self.hmax, h1*self.SF*err**(-0.2))
                else:
                    h2 = max(self.hmax, h1*self.SF*err**(-0.2))
                break
            
        x1 = x + h1
        return x1, y5, h2