import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy import constants
from astropy import constants as con

#region Calculation of coord.
def rotate(origin, point, angle):
    x = origin[0]
    y = origin[1]
    px = point[0]
    py = point[1]
    nx = x + np.cos(angle) * (px - x) - np.sin(angle) * (py - y)
    ny = y + np.sin(angle) * (px - x) + np.cos(angle) * (py - y)
    return np.array([nx, ny])

def distance(x1, x2, y1, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def EqOfMovement(y,t,mue,beta,alpha):
    
    #  y[0,1,2,3] = phi,psi,phi',psi'   y[0,1,2,3] = r,theta,r',theta'
    # dy[0,1,2,3] = phi',psi',phi'',psi''   dy[0,1,2,3] = r',theta',r'',theta''
    
    dy = np.zeros_like(y)
    dy[0] = y[2]; dy[1] = y[3]; 
    #radial_acceleration = beta*(mue/(r**2)) * (np.cos(alpha))**3
    #tan_acceleration = beta*(mue/(r**2)) * (np.cos(alpha))**2 * np.sin(alpha)
    #default alpha = np.pi / 2
    
    a11 = 1; a12 = 0
    b1 = y[0] * (y[3])**2 - mue/(y[0]**2)
    
    a21 = 0; a22 = 1
    b2 = (-2 * y[2] * y[3])/y[0]

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a21*a12)
    dy[3] = (a11*b2 - a21*b1)/(a11*a22 - a21*a12)
    
    return dy

t0 = 0; 
y0 = [0,(np.pi)/18,0,0]
#y0 = [(np.pi/3),(np.pi/4),0,0]
t_fin = 1.728e+7; Nt = 2000000
t = np.linspace(t0, t_fin, Nt)  #time grid

#          (m1,m2,a,b,l0,c,g) - start params
#t0 = 0; y0 = [0,(np.pi)/18,0,0];
mue = constants.G * con.GM_sun.value; beta = 0.17; alpha = np.pi/2
params_0 = (mue, beta, alpha)


Y = odeint(EqOfMovement, y0, t ,params_0); print(f'{Y}------------------------------')

r = Y[:, 0]; theta = Y[:, 1]; dr = Y[:, 2]; dtheta = Y[:, 3]
ddr = np.array([EqOfMovement(yi,ti,mue, beta, alpha)[2] for yi,ti in zip(Y,t)])
#endregion  

#region Animation
#    (m1,m2,a,b,l0,c,g) - start params
x0, y0 = 0, 0

Earth_R = 149.6e+6 
xC, yC = (x0 + Earth_R*np.cos(theta)), (y0 + Earth_R*np.sin(theta))

fig = plt.figure(figsize=[13,9])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
solarSYS_R = 4.5e+8
ax.set(xlim=[-solarSYS_R,solarSYS_R],ylim=[-solarSYS_R,solarSYS_R])


C = ax.plot(xC, yC, 'o', color='green')[0]

def kadr(i):
   
    C.set_data(xC,yC)   
    print(f'itter:{i}, coord:{xC},{yC}')
    return [C]
            
kino = FuncAnimation(fig, kadr, interval = t[1]-t[2], frames=len(t))


#endregion 

#region Graphs of Coord.
fig0 = plt.figure(figsize=[13,9])

ax1 = fig0.add_subplot(2,2,1)
ax1.plot(t,r,color=[1,0,0])
ax1.set_title('r(t)')

ax2 = fig0.add_subplot(2,2,2)
ax2.plot(t,theta,color=[0,1,0])
ax2.set_title('theta(t)')

ax3 = fig0.add_subplot(2,2,3)
ax3.plot(t,dr,color=[0,0,1])
ax3.set_title('dr(t)')

ax4 = fig0.add_subplot(2,2,4)
ax4.plot(t,dtheta,color=[0,0,0])
ax4.set_title('dtheta(t)')

plt.show()
#endregion    