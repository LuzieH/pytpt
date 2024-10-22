import sampling
from plotting import plot_3well_potential, plot_3well_vector_field

import numpy as np

import os.path

# paths
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
figures_path = os.path.join(my_path, 'figures')
example_name = 'triplewell'

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

# triple well in 2D gradient dV plus circular forcing
M = 6 # length of period

# forcing is the vector field sin(t)*f[(-y,x)], where
# f applies some convolution, such that
factor_forced = 1.4
dV_forced = lambda x, y, m: np.array([dV_param_x(x, y, 3), dV_param_y(x, y, 3)]) \
                + factor_forced*np.cos(m*2.*np.pi/M)* np.array([-y, x])

# plot potential and gradient
title = 'Triple well Potential'
subtitles=[
    r'$V(x, y)$',
]
plot_3well_potential(
    potential=V0,
    title=title,
    file_path=os.path.join(figures_path, example_name + '_' + 'potential.png'),
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
    title=title,
    file_path=os.path.join(figures_path, example_name + '_' \
                           + 'vector_field.png'),
    subtitles=subtitles,
)

#count matrix (triple well, no extra forcing)
interval = np.array([[-2, 2], [-1.2, 2.2]]) #size of state space

# discretization of state space into dx cells for transition matrix
dx_power = 1
dx = 2. / (10**dx_power)

# box centers in x and y direction
x = np.arange(interval[0, 0], interval[0, 1] + dx, dx)
y = np.arange(interval[1, 0], interval[1, 1] + dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] # discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim * ydim # dimension of the statespace
xn = np.reshape(xv, (xdim*ydim, 1))
yn = np.reshape(yv, (xdim*ydim, 1))

grid = np.squeeze(np.array([xn, yn]))

# row stochastic transition matrix
T = sampling.transitionmatrix_2D(
    force=dV0,
    sigma=1.0, # lag time of transition matrix is lag*dt
    dt=0.02, # dt for Euler Maruyama discretization
    lag=15,
    Nstep=10000, # number of seeds per box for count matrix
    interval=interval,
    x=x,
    y=y,
    dx=dx,
)

# row stochastic transition matrix
T_small_noise = sampling.transitionmatrix_2D(
    force=dV0,
    sigma=0.26,
    dt=0.02,
    lag=15,
    Nstep=40000,
    interval=interval,
    x=x,
    y=y,
    dx=dx,
)

sigma=1.0
# transition matrix for triple well plus circular forcing
T_m = np.zeros((M, dim_st, dim_st))
for m in np.arange(M):
    T_m[m, :, :] = sampling.transitionmatrix_2D(lambda x, y : \
         dV_forced(x, y, m), sigma=1.0, dt=0.02, lag=15, Nstep=10000, interval=interval, x=x, y=y, dx=dx )

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


# save construction
npz_path = os.path.join(data_path, example_name + '_' + 'construction.npz')
np.savez(
    npz_path,
    interval=interval,
    dx=dx,
    ind_A=ind_A,
    ind_B=ind_B,
    ind_C=ind_C,
    T=T,
    T_m=T_m,
    T_small_noise=T_small_noise,
)
