from pygro.integrators import *
import matplotlib.pyplot as plt

def f(x, y):
    yy, vy = y
    return np.array([vy, -np.sin(yy)])

ck = CashKarp(f)

x0 = 0
y0 = np.array([np.pi/3,0])
x1 = 40

'''
x_s = np.linspace(x0, x1, 500)
y_s = 2*np.exp(np.sin(x_s))-1
'''

y = ck.integrate(x0, x1, y0, 0.0001, 1e-10, interpolate = True)

#plt.plot(x, y)
#plt.plot(x_s, y_s)

#plt.show()