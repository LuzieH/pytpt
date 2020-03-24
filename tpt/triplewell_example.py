import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

from plotting import plot_3well, \
                     plot_3well_effcurrent, \
                     plot_rate, \
                     plot_reactiveness
import numpy as np
import matplotlib.pyplot as plt

import os.path
 
# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
charts_path = os.path.join(my_path, 'charts')
example_name = 'triplewell'

# load triple well construction data
triplewell_construction = np.load(
    os.path.join(data_path, example_name + '_' + 'construction.npz'),
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

#define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A] = 1
densAB[ind_B] = 1

#############################################################################
# infinite-time ergodic

# instantiate
well3 = tp.transitions_mcs(T, ind_A, ind_B, ind_C)
stat_dens = well3.stationary_density()

# compute committor probabilities
[q_f, q_b] = well3.committor()

# therof compute the normalized reactive density
reac_norm_factor = well3.reac_norm_factor()
norm_reac_dens = well3.norm_reac_density()

# and reactive currents
[current, eff_current] = well3.reac_current()
rate = well3.transition_rate()  # AB discrete transition rate

mean_length = well3.mean_transition_length()

#############################################################################
# periodic
M = np.shape(T_m)[0]

def Tm(m): 
    return T_m[np.mod(m,M),:,:].squeeze()

# instantiate
well3_periodic = tpp.transitions_periodic(Tm, M, ind_A, ind_B, ind_C)
stat_dens_p = well3_periodic.stationary_density()

[q_f_p, q_b_p] = well3_periodic.committor()
P_back_m = well3_periodic.backward_transitions()

# normalized reactive density
reac_norm_factor_p = well3_periodic.reac_norm_factor()
norm_reac_dens_p = well3_periodic.norm_reac_density()

# and reactive currents
[current_p, eff_current_p] = well3_periodic.reac_current()
 
[rate_p, time_av_rate_p] = well3_periodic.transition_rate()

mean_length_p = well3_periodic.mean_transition_length()

###################################
#finite-time

def Tn(n):  
    return T#T_m[np.mod(m,M),:,:].squeeze()

N = 6 #time window

# initial density
init_dens_triple = stat_dens

# instantiate
well3_finite = tpf.transitions_finite_time(Tn, N, ind_A, ind_B,  ind_C, init_dens_triple)
 
dens_f = well3_finite.density()
[q_f_f, q_b_f] = well3_finite.committor()
 
# normalized reactive density
reac_norm_factor_f = well3_finite.reac_norm_factor()
norm_reac_dens_f = well3_finite.norm_reac_density()

# and reactive currents
[current_f, eff_current_f] = well3_finite.reac_current()

[rate_f, time_av_rate_f] = well3_finite.transition_rate()

mean_length_f = well3_finite.mean_transition_length()


print("rate (infinite-time, stationary): %f" % rate)
print("periodic-averaged rate (infinite-time, periodic): %f" % time_av_rate_p[0])
print("time-averaged rate (finite-time, time-homogeneous): %f" % time_av_rate_f[0])

print("mean length (infinite-time, stationary): %f" % mean_length)
print("mean length (infinite-time, periodic): %f" % mean_length_p)
print("mean length (finite-time, time-homogeneous): %f" % mean_length_f)



###################################
# finite time bifurcation analysis 

#finite-time

def Tn_small_noise(n):  
    return T_small_noise#T_m[np.mod(m,M),:,:].squeeze()

# compute stationary density of triple well with small noise to get initial density
well3_small_noise = tp.transitions_mcs(T_small_noise, ind_A, ind_B, ind_C)
stat_dens_small_noise = well3_small_noise.stationary_density()

init_dens_triple_bif = stat_dens_small_noise

#N_bif_array = np.array([20, 50, 100, 500])#time window 20-> lower channel only in stat dens, time window 50, lower channel in both
N_bif_size = np.shape(N_bif_array)[0]

norm_reac_dens_f_bif_all = np.zeros((N_bif_size,dim_st))
eff_current_f_bif_all = np.zeros((N_bif_size,dim_st,2))  
color_current_f_bif_all = np.zeros((N_bif_size,dim_st))

subtitles_bif_dens = []
subtitles_bif_eff = []

ind = 0
for N_bif in N_bif_array:
    
    # instantiate
    well3_finite_bif = tpf.transitions_finite_time(Tn_small_noise, N_bif, ind_A, ind_B,  ind_C, init_dens_triple_bif)
     
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
npz_path = os.path.join(data_path, example_name + '_' + 'ergodic.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens,
    q_f=q_f,
    q_b=q_b,
    norm_reac_dens=norm_reac_dens,
    eff_current=eff_current,
    rate=rate,
)
npz_path = os.path.join(data_path, example_name + '_' + 'periodic.npz')
np.savez(
    npz_path,
    stat_dens=stat_dens_p,
    q_f=q_f_p,
    q_b=q_b_p,
    norm_reac_dens=norm_reac_dens_p,
    reac_norm_factor=reac_norm_factor_p,
    eff_current=eff_current_p,
    rate=rate_p,
    time_av_rate=time_av_rate_p,
)
npz_path = os.path.join(data_path, example_name + '_' + 'finite.npz')
np.savez(
    npz_path,
    stat_dens=dens_f,
    q_f=q_f_f,
    q_b=q_b_f,
    norm_reac_dens=norm_reac_dens_f,
    reac_norm_factor=reac_norm_factor_f,
    eff_current=eff_current_f,
    rate=rate_f,
    time_av_rate=time_av_rate_f,
)
npz_path = os.path.join(data_path, example_name + '_' + 'bifurcation.npz')
np.savez(
    npz_path,
    norm_reac_dens=norm_reac_dens_f_bif_all,
    eff_current=eff_current_f_bif_all,
    color_current=color_current_f_bif_all,
)
