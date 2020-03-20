import transition_matrix_from_samples as tms
from plotting import plot_3well_potential, plot_3well_vector_field

import numpy as np

import os.path

my_path = os.path.abspath(os.path.dirname(__file__))


##############################################################################
#triple well in 2D energy landscape V and gradient dV
factor = 0.25
V_param = lambda x, y, p: -1 * factor*(3*np.exp(-x**2-(y-(1./3))**2) \ 
                - p*np.exp(-x**2-(y-(5./3))**2) - 5*np.exp(-(x-1)**2-y**2) \
                - 5*np.exp(-(x+1)**2-y**2)  + 0.2*(x**4) + 0.2*(y-1./3)**4)

dV_param_x = lambda x, y, p: -1 * factor*((-2*3*x)*np.exp(-x**2-(y-(1./3))**2) \
                + (p*2*x)*np.exp(-x**2-(y-(5./3))**2) \
                + (10*(x-1))*np.exp(-(x-1)**2-y**2) \
                + (10*(x+1))*np.exp(-(x+1)**2-y**2)  + 0.8*(x**3))
dV_param_y = lambda x, y, p: -1 * factor*((-2*3*(y-1./3)) \
                * np.exp(-x**2-(y-(1./3))**2) \
                + (p*2*(y-(5./3)))*np.exp(-x**2-(y-(5./3))**2) \
                + (10*y)*np.exp(-(x-1)**2-y**2) \
                + (10*y)*np.exp(-(x+1)**2-y**2)  + 0.8*(y-1./3)**3)

V0 = lambda x, y: -1 * V_param(x, y, 3)
dV0 = lambda x, y: np.array([dV_param_x(x, y, 3), dV_param_y(x, y, 3)])

##############################################################################
# triple well in 2D gradient dV plus circular forcing
M = 6 # length of period

# forcing is the vector field sin(t)*f[(-y,x)], where 
# f applies some convolution, such that 
factor_forced = 1.4
dV_forced = lambda x, y, m: np.array([dV_param_x(x, y, 3), dV_param_y(x, y, 3)]) \
                + factor_forced*np.cos(m*2.*np.pi/M)* np.array([-y, x])

charts_path = os.path.join(my_path, 'charts')
example_name = 'triplewell'
title = 'Triple well Potential'
subtitles=[
    r'$V(x, y)$', 
]
plot_3well_potential(
    potential=V0,
    file_path=os.path.join(charts_path, example_name + '_' + 'potential.png'),
    title=title,
    subtitles=subtitles,
)
title = 'Triple well Gradient and Force'
subtitles=[
    r'$-\nabla V(x, y)$', 
    r'$-\nabla V(x, y) + F(0, x, y)$', 
    r'$-\nabla V(x, y) + F(3, x, y)$', 
]
plot_3well_vector_field(
    vector_field=dV0,
    vector_field_forced=dV_forced,
    file_path=os.path.join(charts_path, example_name + '_' \ 
                           + 'vector_field.png'),
    title=title,
    subtitles=subtitles,
)

##############################################################################
#count matrix (triple well, no extra forcing)
interval = np.array([[-2, 2], [-1.2, 2.2]]) #size of state space
dim = np.shape(interval)[0] #dimension of state space
# discretization of state space into dx cells for transition matrix
dx_power = 1 
dx = 2./(10**dx_power)
# box centers in x and y direction
x = np.arange(interval[0, 0], interval[0, 1]+dx, dx) 
y = np.arange(interval[1, 0], interval[1, 1]+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] # discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim * ydim # dimension of the statespace
xn = np.reshape(xv, (xdim*ydim, 1))
yn = np.reshape(yv, (xdim*ydim, 1))

grid = np.squeeze(np.array([xn, yn]))

Nstep = 10000 # number of seeds per box for count matrix
sigma = 1.0 
dt= 0.02 # dt for Euler Maruyama discretization
lag= 15 # lag time of transition matrix is lag*dt

# row stochastic transition matrix
T = tms.transitionmatrix_2D(dV0, sigma, dt, lag, Nstep, interval, x, y, dx, dim)

# row stochastic transition matrix
sigma_small = 0.26
T_small_noise=tms.transitionmatrix_2D(dV0, sigma_small, dt, lag, 4 * Nstep, \
                                      interval, x, y, dx, dim)

##############################################################################
# transition matrix for triple well plus circular forcing
T_m = np.zeros((M, dim_st, dim_st))
for m in np.arange(M):
    T_m[m, :, :] = tms.transitionmatrix_2D(lambda x, y : dV_forced(x, y, m), sigma, \ 
                                           dt, lag, Nstep, interval, x, y, dx, dim)

##############################################################################
# defining A and B
# define by center and radius!
A_center = np.array([-1, 0])  
B_center = np.array([1, 0])
radius_setAB = 0.425  

def set_A_triplewell(x, A_center, radius_setAB):
    if np.linalg.norm(x-A_center) <= radius_setAB:
        return 1
    else:
        return 0
    
def set_B_triplewell(x, B_center, radius_setAB):
    if np.linalg.norm(x-B_center) <= radius_setAB:
        return 1
    else:
        return 0
    
def set_C_triplewell(x, A_center, B_center, radius_setAB):
    if (set_A_triplewell(x,A_center, radius_setAB) == 0 and
        set_B_triplewell(x,B_center, radius_setAB) == 0):
        return 1
    else:
        return 0

# indices of transition region C
ind_C = np.argwhere(
        np.array([
            set_C_triplewell(grid[:, i], A_center, B_center,radius_setAB)
            for i in np.arange(np.shape(xn)[0])
        ])==1
    ).squeeze()

# indices of B
ind_B = np.argwhere(
        np.array([
            set_B_triplewell(grid[:, i], B_center, radius_setAB)
            for i in np.arange(np.shape(xn)[0])
        ])==1
    ).flatten()

# indices of B
ind_A = np.argwhere(
        np.array([
            set_A_triplewell(grid[:, i], A_center, radius_setAB)
            for i in np.arange(np.shape(xn)[0])])==1
        ).flatten()
 
 
############################################################################## 
# save
np.save(os.path.join(my_path, 'data/triplewell_T.npy'), T)
np.save(os.path.join(my_path, 'data/triplewell_T_m.npy'), T_m)
np.save(os.path.join(my_path, 'data/triplewell_T_small_noise.npy'), \
        T_small_noise)
np.save(os.path.join(my_path, 'data/triplewell_interval.npy'), interval)
np.save(os.path.join(my_path, 'data/triplewell_dx.npy'), dx)
np.save(os.path.join(my_path, 'data/triplewell_ind_A.npy'), ind_A)
np.save(os.path.join(my_path, 'data/triplewell_ind_B.npy'), ind_B)
np.save(os.path.join(my_path, 'data/triplewell_ind_C.npy'), ind_C)
