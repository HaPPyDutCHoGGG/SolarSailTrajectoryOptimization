from scipy.integrate import odeint
import scipy.constants as constants
import numpy as np
import matplotlib.pyplot as plt

M = 1.989 * (10 ** 30)
G = constants.G
alpha = 30
alpha0 = (alpha / 180) * np.pi
v00 = 0.7  # km/s
v0 = v00 * 1000  # m/s


def dqdt(q, t):
    x = q[0]
    y = q[2]
    ax = - G * M * (x / ((x ** 2 + y ** 2) ** 1.5))
    ay = - G * M * (y / ((x ** 2 + y ** 2) ** 1.5))
    return [q[1], ax, q[3], ay]

vx0 = v0 * np.cos(alpha0)
vy0 = v0 * np.sin(alpha0)
x0 = -1.5 * (10 ** 11)
y0 = 0 * (10 ** 11)
q0 = [x0, vx0, y0, vy0]

N = 1000000
t = np.linspace(0.0, 100000000000.0, N)

pos = odeint(dqdt, q0, t)

x1 = pos[:, 0]
y1 = pos[:, 2]

plt.plot(x1, y1, 0, 0, 'ro')
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True)
plt.show()