import numpy as np
from astropy import constants
from scipy.integrate import solve_ivp
from collections import namedtuple
import matplotlib.pyplot as plt





def dqdt(t, q, p):
    # Функция правых частей
    # t - текущее время
    # q - вектор состояния системы
    # p - параметры системы 
    r = q[0]
    vr   = q[1]
    theta   = q[2]
    vtheta  = q[3]
    
    mue, beta, alpha = list(p)
    
    # Коэффициенты при ускорения в системе линейных уравнений
    # относительно производных
    a11 = 1
    a12 = 0
    
    a21 = 0
    a22 = r

    ar = beta * (mue/(r**2)) * (np.cos(alpha))**3
    atheta = beta * (mue/(r**2)) * (np.cos(alpha))**2 * np.sin(alpha)
    
    # Элементы матрицы правых частей
    b1  = r * (vtheta)**2 - (mue/(r**2)) + ar
    b2  = -2*(vr * vtheta) + atheta
    
    # Составляем матрицы системы линейный уравнений относительно вторых производных
    A   = np.array( [[a11, a12], [a21, a22]] )
    B   = np.array( [b1,b2])

    # Решаем систему линейных уравнений
    d2q = np.linalg.solve(A, B)   
    
    # Получаем радиальное ускорение 
    d2r = d2q[0]
    # и угловое относительное солнца
    d2theta   = d2q[1]
    
    # q содержит координату и скорость,
    # возвращаем производную от q - скорость и ускорение    
    return (vr, d2r, vtheta, d2theta)

mue = constants.M_sun.value * constants.G.value
beta = 0.17
alpha = np.pi / 3
year_sec = 60*60*24*365*1.5
V_EartORB = 3e+4 #constants.au.value  per AU
W_EarthORB = V_EartORB/constants.au.value 
# Создаем структуру данных 
Parameters = namedtuple('Parameters', 'mue,beta,alpha')

# Создаем набор параметров
p = Parameters(mue,beta,alpha)

#q0  = [1, np.pi*(1/4), V_EartORB, ]
q0  = [constants.au.value, 0, np.pi*(1/2), W_EarthORB]
sol = solve_ivp(lambda t, q: dqdt(t, q, p), [0, year_sec], q0, rtol=1e-10)#rtol = 1e-7
print(sol)
print(len(sol.y[0]))

_r = sol.y[0]

_theta = sol.y[2]

ax = plt.subplot(111,projection='polar')
ax.plot(_theta,_r,color='r',linewidth=3)
#ax.scatter([],[])
ax.grid(True)

plt.show()


