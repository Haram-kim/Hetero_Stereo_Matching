"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""

import numpy as np
from scipy.linalg import expm, logm

def SO3(w):
    return expm(hat(w))

def hat(x):
    x_hat = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return x_hat

def SE3(r, t = None):
    if t is None:
        xi = r
    else:
        xi = np.concatenate([t, r],0)
    se3 = np.zeros((4,4))
    se3[:3,:3] = hat(xi[3:])
    se3[:3, 3] = xi[:3]
    return expm(se3)

def InvSE3(SE3):
    xi = np.zeros(6)
    se3 = logm(SE3)

    xi[:3] = se3[:3, 3]
    xi[3] = se3[2,1]
    xi[4] = se3[0,2]
    xi[5] = se3[1,0]

    return xi


