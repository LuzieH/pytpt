import numpy as np
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))



##############################################################################
#triple well in 2D, energy landscape V and gradient dV

factor=0.25
V_param = lambda x,y,p: factor*(3*np.exp(-x**2-(y-(1./3))**2) - p*np.exp(-x**2-(y-(5./3))**2) - 5*np.exp(-(x-1)**2-y**2) - 5*np.exp(-(x+1)**2-y**2)  + 0.2*(x**4) + 0.2*(y-1./3)**4)

V0 = lambda x,y: V_param(x,y,3)

dV_param_x = lambda x,y,p: factor*((-2*3*x)*np.exp(-x**2-(y-(1./3))**2) +(p*2*x)*np.exp(-x**2-(y-(5./3))**2) + (10*(x-1))*np.exp(-(x-1)**2-y**2) + (10*(x+1))*np.exp(-(x+1)**2-y**2)  + 0.8*(x**3))
dV_param_y = lambda x,y,p: factor*((-2*3*(y-1./3))*np.exp(-x**2-(y-(1./3))**2) + (p*2*(y-(5./3)))*np.exp(-x**2-(y-(5./3))**2) + (10*y)*np.exp(-(x-1)**2-y**2) + (10*y)*np.exp(-(x+1)**2-y**2)  + 0.8*(y-1./3)**3)

dV0 = lambda x,y: np.array([dV_param_x(x,y,3),dV_param_y(x,y,3)])


##############################################################################
#count matrix (triple well, no extra forcing)

interval = np.array([[-2,2],[-2,3]]) #size of state space
dim=np.shape(interval)[0] #dimension of state space
dx_power=1 
dx=1./(10**dx_power) #discretization of state space into dx cells for transition matrix
x = np.arange(interval[0,0],interval[0,1]+dx, dx) #box centers in x and y direction
y = np.arange(interval[1,0],interval[1,1]+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] #discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim*ydim # dimension of the statespace
xn=np.reshape(xv,(xdim*ydim,1))
yn=np.reshape(yv,(xdim*ydim,1))

grid = np.squeeze(np.array([xn,yn]))

Nstep=100 #number of seeds per box for count matrix
sigma=1.0 
dt=0.01 #dt for Euler Maruyama discretization
lag=10 #lag time of transition matrix is lag*dt

#row stochastic transition matrix
T=transitionmatrix_2D(dV3,sigma,dt, lag ,Nstep, interval, dx_power, x, y, dim)

############################################################################
#transition matrix for triple well plus circular forcing
M=6 #length of period

#forcing is the vector field sin(t)*f[(-y,x)], where f applies some convolution, such that 
factor_forced=0.5
dV_forced = lambda x,y,m: np.array([dV_param_x(x,y,3),dV_param_y(x,y,3)]) +factor_forced*np.cos(m*2.*np.pi/M)* np.array([-y,x])

T_m =np.zeros((M, dim_st, dim_st))
for m in np.arange(M):
    T_m[m,:,:]= transitionmatrix_2D(lambda x,y : dV_forced(x,y,m) ,sigma,dt, lag ,Nstep, interval, dx_power, x, y, dim)




##############################################################################
# defining A and B
#define by center and radius!
A_center = np.array([-1,0])  
B_center = np.array([1,0])
radius_setAB = 0.4

def set_A_triplewell(x,A_center, radius_setAB):
    if np.linalg.norm(x-A_center) <=radius_setAB:
        return 1
    else:
        return 0
    
def set_B_triplewell(x,B_center, radius_setAB):
    if np.linalg.norm(x-B_center)<=radius_setAB:
        return 1
    else:
        return 0
    
def set_C_triplewell(x,A_center, B_center, radius_setAB):
    if set_A_triplewell(x,A_center, radius_setAB) ==0 and set_B_triplewell(x,B_center, radius_setAB) ==0:
        return 1
    else:
        return 0

#indices of transition region C
ind_C = np.array([set_C_triplewell(grid[:,i],A_center,B_center,radius_setAB) for i in np.arange(np.shape(xn)[0])])

#indices of B
ind_B = np.array([set_B_triplewell(grid[:,i],B_center, radius_setAB) for i in np.arange(np.shape(xn)[0])])

#indices of B
ind_A = np.array([set_A_triplewell(grid[:,i],A_center, radius_setAB) for i in np.arange(np.shape(xn)[0])])

##############################################################################

#stat dens

# compute stationary density
eigv, eig = np.linalg.eig(np.transpose(T))
# get index of eigenvector with eigenvalue 1 (up to small numerical error)
index = np.where(np.isclose(eigv, 1))[0]
# normalize
stat_dens = (np.real(eig[:, index]) /
             np.sum(np.real(eig[:, index]))).flatten()
plt.imshow(np.real(stat_dens).reshape((51, 41)))


##################################################################################
# save

np.save(os.path.join(my_path, 'data/triplewell_T.npy'), T)
np.save(os.path.join(my_path, 'data/triplewell_T_m.npy'), T_m)
np.save(os.path.join(my_path, 'data/triplewell_interval.npy'), interval)
np.save(os.path.join(my_path, 'data/triplewell_dx.npy'), dx)
np.save(os.path.join(my_path, 'data/triplewell_ind_A.npy'), ind_A)
np.save(os.path.join(my_path, 'data/triplewell_ind_B.npy'), ind_B)
np.save(os.path.join(my_path, 'data/triplewell_ind_C.npy'), ind_C)