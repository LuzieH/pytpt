from pytpt import stationary  
from pytpt import periodic  
from pytpt import finite  

 
import numpy as np

import os.path

# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')

# load triple well construction data
triplewell_construction = np.load(
    os.path.join(data_path, 'triplewell_construction.npz'),
    allow_pickle=True,
)
interval = triplewell_construction['interval']
dx = triplewell_construction['dx']
ind_A = triplewell_construction['ind_A']
ind_B = triplewell_construction['ind_B']
ind_C = triplewell_construction['ind_C']
T = triplewell_construction['T']
T_m = triplewell_construction['T_m']
T_small_noise = triplewell_construction['T_small_noise']

#state space
x = np.arange(interval[0,0], interval[0,1] + dx, dx) #box centers in x and y direction
y = np.arange(interval[1,0], interval[1,1] + dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] #discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim * ydim # dimension of the statespace
xn = np.reshape(xv, (xdim * ydim, 1))
yn = np.reshape(yv, (xdim * ydim, 1))
grid = np.squeeze(np.array([xn, yn]))


# TPT ergodic, infinite-time
example_name = 'triplewell_stationary'
# instantiate
well3 = stationary.tpt(T, ind_A, ind_B, ind_C)
# compute statistics
well3.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
well3.save_statistics(npz_path)


# TPT periodic
example_name = 'triplewell_periodic'
M = np.shape(T_m)[0]

def Tm(m): 
    return T_m[np.mod(m, M), :, :].squeeze()

# instantiate
well3_periodic = periodic.tpt(Tm, M, ind_A, ind_B, ind_C)
# compute statistics
well3_periodic.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
well3_periodic.save_statistics(npz_path)


# TPT finite time, time-homogeneous
example_name = 'triplewell_finite'

def Tn(n):  
    return T #T_m[np.mod(m,M),:,:].squeeze()

N = 6 #time window

# initial density
init_dens_well3_finite = well3._stat_dens
# instantiate
well3_finite = finite.tpt(
    Tn,
    N,
    ind_A,
    ind_B,
    ind_C,
    init_dens_well3_finite,
)
# compute statistics
well3_finite.compute_statistics()
# save statistics
npz_path = os.path.join(data_path, example_name + '.npz')
well3_finite.save_statistics(npz_path)
 


# finite time bifurcation analysis 
example_name = 'triplewell_bifurcation'

def Tn_small_noise(n):  
    return T_small_noise # T_m[np.mod(m,M),:,:].squeeze()

# compute stationary density of triple well with small noise to get initial density
well3_small_noise = stationary.tpt(T_small_noise, ind_A, ind_B, ind_C)
stat_dens_small_noise = well3_small_noise.stationary_density()
init_dens_triple_bif = stat_dens_small_noise

#time window 20-> lower channel only in stat dens, time window 50, lower channel in both
N_bif_array = np.array([20, 50, 100, 500])
N_bif_size = np.shape(N_bif_array)[0]

norm_reac_dens_f_bif_all = np.zeros((N_bif_size, dim_st))
eff_current_f_bif_all = np.zeros((N_bif_size, dim_st, 2))  
color_current_f_bif_all = np.zeros((N_bif_size, dim_st))

subtitles_bif_dens = []
subtitles_bif_eff = []

ind = 0
for N_bif in N_bif_array:
    
    # instantiate
    well3_finite_bif = finite.tpt(
        Tn_small_noise,
        N_bif,
        ind_A,
        ind_B,
        ind_C,
        init_dens_triple_bif,
    )
     
    dens_f_bif = well3_finite_bif.density()
    [q_f_f_bif, q_b_f_bif] = well3_finite_bif.committor()
     
    # normalized reactive density
    reac_norm_factor_f_bif = well3_finite_bif.reac_norm_factor()
    norm_reac_dens_f_bif = well3_finite_bif.norm_reac_density()
    
    # and reactive currents
    [current_f_bif, eff_current_f_bif] = well3_finite_bif.reac_current()
    
    [rate_f_bif, time_av_rate_f_bif] = well3_finite_bif.transition_rate()
    
    norm_reac_dens_f_bif_all[ind,:] = norm_reac_dens_f_bif[int(N_bif/2)]
    
    #calculation the effective vector for each state
    eff_vectors_f_bif = np.zeros((dim_st, 2))
    eff_vectors_unit_f_bif = np.zeros((dim_st, 2))
    colors_f_bif = np.zeros(dim_st)
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]])) > 0:
                eff_vectors_f_bif[i, 0] += eff_current_f_bif[int(N_bif/2),i,j] *  (xn[j] - xn[i]) *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_f_bif[i, 1] += eff_current_f_bif[int(N_bif/2),i,j] *  (yn[j] - yn[i]) *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]])))  
        colors_f_bif[i] = np.linalg.norm(eff_vectors_f_bif[i, :])
        if colors_f_bif[i] > 0:
            eff_vectors_unit_f_bif[i, :] = eff_vectors_f_bif[i, :]/colors_f_bif[i] 
            
            
    eff_current_f_bif_all[ind, :, :] = eff_vectors_unit_f_bif
    color_current_f_bif_all[ind, :] = colors_f_bif
    
    subtitles_bif_dens.append('$\hat{\mu}^{AB}$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
    subtitles_bif_eff.append('$f^+$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
    
    ind = ind + 1

# save the transition statistics in npz files
npz_path = os.path.join(data_path, example_name + '.npz')
np.savez(
    npz_path,
    norm_reac_dens=norm_reac_dens_f_bif_all,
    eff_current=eff_current_f_bif_all,
    color_current=color_current_f_bif_all,
)
