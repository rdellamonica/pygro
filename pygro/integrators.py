import numpy as np
import scipy.interpolate as sp_int

class Integrator():
    def __init__(self):
        pass

class CashKarp(Integrator):
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
            w = v/(self.Atol+self.Rtol*v)
            return np.linalg.norm(w)**(1/(1+i))
        else:
            v = y1-y2
            return abs(v/(self.Atol+self.Rtol*v))**(1/(1+i))

def dp4(self, x, u, tau, h, Atol, Rtol):
    rho = 0
    h1 = h
    while rho < 1:
        k1 = h1*self.motion_eq(x, u)
        k2 = h1*self.motion_eq(x+k1[0]/5, u+k1[1]/5)
        k3 = h1*self.motion_eq(x+(3*k1[0]/40+9*k2[0]/40), u+(3*k1[1]/40+9*k2[1]/40))
        k4 = h1*self.motion_eq(x+(44/45*k1[0]-56/15*k2[0]+32/9*k3[0]), u+(44/45*k1[1]-56/15*k2[1]+32/9*k3[1]))
        k5 = h1*self.motion_eq(x+(19372/6561*k1[0]-25360/2187*k2[0]+64448/6561*k3[0]-212/729*k4[0]), u+(19372/6561*k1[1]-25360/2187*k2[1]+64448/6561*k3[1]-212/729*k4[1]))
        k6 = h1*self.motion_eq(x+(9017/3168*k1[0]-355/33*k2[0]+46732/5247*k3[0]+49/176*k4[0]-5103/18656*k5[0]), u + (9017/3168*k1[1]-355/33*k2[1]+46732/5247*k3[1]+49/176*k4[1]-5103/18656*k5[1]))
        k7 = h1*self.motion_eq(x+(35/384*k1[0]+500/1113*k3[0]+125/192*k4[0]-2187/6784*k5[0]+11/84*k6[0]), u + (35/384*k1[1]+500/1113*k3[1]+125/192*k4[1]-2187/6784*k5[1]+11/84*k6[1]))
        
        S0 = 35/384*k1[0]+500/1113*k3[0]+125/192*k4[0]-2187/6784*k5[0]+11/84*k6[0]
        S1 = 35/384*k1[1]+500/1113*k3[1]+125/192*k4[1]-2187/6784*k5[1]+11/84*k6[1]

        S2 = 5179/57600*k1[0] + 7571/16695*k3[0] + 393/640*k4[0] -92097/339200*k5[0]+187/2100*k6[0]+1/40*k7[0]
        S3 = 5179/57600*k1[1] + 7571/16695*k3[1] + 393/640*k4[1] -92097/339200*k5[1]+187/2100*k6[1]+1/40*k7[1]
        
        x5 = x + S0
        u5 = u + S1

        x4 = x + S2
        u4 = u + S3

        tol = []

        for i in range(4):
            tol_i = Atol+max(abs(x5[i]),abs(x[i]))*Rtol
            tol.append(tol_i)
        
        for i in range(4):
            tol_i = Atol+max(abs(u5[i]),abs(u[i]))*Rtol
            tol.append(tol_i)
        
        epsilon = 0
        for i in range(4):
            epsilon += ((x5[i]-x4[i])/(tol[i]))**2
            epsilon += ((u5[i]-u4[i])/(tol[i+4]))**2
        epsilon = (epsilon/4)**0.5

        if epsilon != 0:
            h1 = h1*min(0.8*(1/epsilon)**(1/5),1.5)
            if epsilon <= 1:
                rho = 1
            else:
                rho = 0
        else:
            rho = 1

    tau1 = tau + h1

    return [x5, u5, tau1, h1]

def fh78(self, x, u, tau, h, Rtol, Atol):
    rho = 0
    h1 = h

    x = np.array(x)
    u= np.array(u)

    delta  = Atol

    while rho < 1:
        
        bb = np.array(
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

        c1 = [41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 41/180, 0, 0]
        c2 = [0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/180, 41/180]

        k_x = np.zeros(13, dtype = object)
        k_u = np.zeros(13, dtype = object)

        for i in range(12):
            k_x[i+1] = h*self.motion_eq(x+np.dot(bb[i], k_x), u +np.dot(bb[i], k_u))[0]
            k_u[i+1] = h*self.motion_eq(x+np.dot(bb[i], k_x), u +np.dot(bb[i], k_u))[1]

        S0 = np.dot(c1, k_x)
        S1 = np.dot(c1, k_u)

        S2 = np.dot(c2, k_x)
        S3 = np.dot(c2, k_u)
        
        x7 = x + S0
        u7 = u + S1

        x8 = x + S2
        u8 = u + S3

        epsilon = np.linalg.norm(np.concatenate((x7,u7))-np.concatenate((x8,u8)), ord = np.inf)

        if epsilon != 0:
            rho = delta/epsilon
            h1 = 0.9*h1*rho**(1/8)
        else:
            rho = 1

    tau1 = tau + h1

    return [x7, u7, tau1, h1]

def dp78(self, x, u, tau, h, Rtol, Atol):
    rho = 0
    h1 = h
    x = np.array(x)
    u= np.array(u)

    delta = Atol

    while rho < 1:
        
        bb = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 1/18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [1/32, 0, 3/32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5/16, 0, -75/64, 75/64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3/80, 0, 0, 3/16, 3/20, 0, 0, 0, 0, 0, 0, 0, 0],
            [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000, 0, 0, 0, 0, 0, 0, 0],
            [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555, 0, 0, 0, 0, 0, 0],
            [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287, 0, 0, 0, 0, 0],
            [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789, 0, 0, 0, 0],
            [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653, 0, 0, 0],
            [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083, 0, 0],
            [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0, 0]
        ])

        c1 = [ 14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731,   561292985/797845732,   -1041891430/1371343529,  760417239/1151165299, 118820643/751138087, -528747749/2220607170,  1/4]
        c2 = [ 13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186,   -3867574721/1518517206,   465885868/322736535,  53011238/667516719,                  2/45,    0]

        k_x = np.zeros(13, dtype = object)
        k_u = np.zeros(13, dtype = object)

        for i in range(12):
            k_x[i+1] = h*self.motion_eq(x+np.dot(bb[i], k_x), u +np.dot(bb[i], k_u))[0]
            k_u[i+1] = h*self.motion_eq(x+np.dot(bb[i], k_x), u +np.dot(bb[i], k_u))[1]

        S0 = np.dot(c1, k_x)
        S1 = np.dot(c1, k_u)

        S2 = np.dot(c2, k_x)
        S3 = np.dot(c2, k_u)
        
        x8 = x + S0
        u8 = u + S1

        x7 = x + S2
        u7 = u + S3

        epsilon = np.linalg.norm(x8-x7, ord = np.inf)

        if epsilon != 0:
            rho = delta/epsilon
            h1 = 0.9*h1*rho**(1/8)
        else:
            rho = 1

    tau1 = tau + h1

    return [x8, u8, tau1, h1]

def bs23(self, x, u, tau, h, Atol, Rtol):

    delta = Atol

    rho = 0
    h1 = h
    while rho < 1:
        k1 = h1*self.motion_eq(x, u)
        k2 = h1*self.motion_eq(x+k1[0]/2, u+k1[1]/2)
        k3 = h1*self.motion_eq(x+(3*k2[0]/4), u+(3*k2[1]/4))
        k4 = h1*self.motion_eq(x+(2/9*k1[0]+1/3*k2[0]+4/9*k3[0]), u+(2/9*k1[1]+1/3*k2[1]+4/9*k3[1]))
        
        S0 = 2/9*k1[0]+1/3*k2[0]+4/9*k3[0]
        S1 = 2/9*k1[1]+1/3*k2[1]+4/9*k3[1]

        S2 = 7/24*k1[0]+1/4*k2[0]+1/3*k3[0]+1/8*k4[0]
        S3 = 7/24*k1[1]+1/4*k2[1]+1/3*k3[1]+1/8*k4[1]
        
        x3 = x + S0
        u3 = u + S1

        x2 = x + S2
        u2 = u + S3

        epsilon = np.linalg.norm(x3-x2, ord = np.inf)

        if epsilon != 0:
            rho = delta/epsilon
            h1 = 0.9*h1*rho**(1/3)
        else:
            rho = 1

    tau1 = tau + h1

    return [x3, u3, tau1, h1]

def rk45(self, x, u, tau, h, Rtol, Atol):
    k1 = self.motion_eq(x, u)
    k2 = self.motion_eq(x+k1[0]*h/2, u+k1[1]*h/2)
    k3 = self.motion_eq(x+k2[0]*h/2, u+k2[1]*h/2)
    k4 = self.motion_eq(x+k3[0]*h,u+k3[1]*h)
    S0 = (k1[0]+2*k2[0]+2*k3[0]+k4[0])*h/6
    S1 = (k1[1]+2*k2[1]+2*k3[1]+k4[1])*h/6
    '''
    if abs(u[1])/x[1] < 1e-4 and S1[1]/x[1] < 1e-4:
        u[1] = 0
        S1[1] = 0
        S0[1] = 0
    '''
    x1 = x + S0
    u1 = u + S1
    tau1 = tau + h
    return [x1, u1, tau1]

def rkf45(self, x, u, tau, h, Atol, Rtol):
    rho = 0
    h1 = h
    while rho < 1:
        y_hh = self.rk45(x, u, tau, h1)
        y_hh = self.rk45(y_hh[0], y_hh[1], y_hh[2], h1)
        y_2h = self.rk45(x, u, tau, 2*h1)
        x_hh = np.array(y_hh[0])
        x_2h = np.array(y_2h[0])
        epsilon = np.linalg.norm(x_hh-x_2h)/30
        if epsilon != 0:
            rho = h1*delta/epsilon
        else:
            rho = 1
        h1 = h1*(rho**0.25)
    
    return [y_hh[0], y_hh[1], y_hh[2], h1]

def step_selector():
    pass