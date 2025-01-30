import numpy as np

def Rotate_x(theta):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return R

def Rotate_y(theta):
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return R

def Rotate_z(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

def cartesian_to_spherical_point(x, y, z):
    r = np.sqrt(x**2+y**2+z**2)
    return r, np.arccos(z/r), np.arctan2(y, x)

def cartesian_to_spherical_vector(x, y, z, vx, vy, vz):
    r = np.sqrt(x**2+y**2+z**2)
    vr = (x*vx+y*vy+z*vz)/r
    vtheta = -((x**2+y**2)*vz-z*(x*vx+y*vy))/(r**2*np.sqrt(x**2+y**2))
    vphi = -(vx*y-x*vy)/(x**2+y**2)
    return vr, vtheta, vphi