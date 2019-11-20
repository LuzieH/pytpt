import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf
 
import numpy as np
import matplotlib.pyplot as plt

import os.path
 
##############################################################################
# load data
 
my_path = os.path.abspath(os.path.dirname(__file__))
T = np.load(os.path.join(my_path, 'data/triplewell_T.npy'))
T_m = np.load(os.path.join(my_path, 'data/triplewell_T_m.npy'))
interval = np.load(os.path.join(my_path, 'data/triplewell_interval.npy'))
dx = np.load(os.path.join(my_path, 'data/triplewell_dx.npy'))
ind_A = np.load(os.path.join(my_path, 'data/triplewell_ind_A.npy'))
ind_B = np.load(os.path.join(my_path, 'data/triplewell_ind_B.npy'))
ind_C = np.load(os.path.join(my_path, 'data/triplewell_ind_C.npy'))

############################################################################
#state space
x = np.arange(interval[0,0],interval[0,1]+dx, dx) #box centers in x and y direction
y = np.arange(interval[1,0],interval[1,1]+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] #discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim*ydim # dimension of the statespace
xn=np.reshape(xv,(xdim*ydim,1))
yn=np.reshape(yv,(xdim*ydim,1))
grid = np.squeeze(np.array([xn,yn]))


#############################################################################
# infinite-time ergodic

# instantiate
well3 = tp.transitions_mcs(T, ind_A, ind_B, ind_C)
stat_dens = well3.stationary_density()

# compute committor probabilities
[q_f, q_b] = well3.committor()

# therof compute the reactive density
reac_dens = well3.reac_density()

# and reactive currents
[current, eff_current] = well3.reac_current()
rate = well3.transition_rate()  # AB discrete transition rate


#############################################################################
# periodic
M=np.shape(T_m)[0]

def Tm(m): 
    return T_m[np.mod(m,M),:,:].squeeze()

# instantiate
well3_periodic = tpp.transitions_periodic(Tm, M, ind_A, ind_B, ind_C)
stat_dens_p = well3_periodic.stationary_density()

[q_f_p, q_b_p] = well3_periodic.committor()
P_back_m = well3_periodic.backward_transitions()
# reactive density
reac_dens_p = well3_periodic.reac_density()

# and reactive currents
[current_p, eff_current_p] = well3_periodic.reac_current()

rate_p = well3_periodic.transition_rate()


###################################
#finite-time

def Tn(n):  
    return T#T_m[np.mod(m,M),:,:].squeeze()

N = 8 #time window

# initial density
init_dens_triple = stat_dens

# instantiate
well3_finite = tpf.transitions_finite_time(Tn, N, ind_A, ind_B,  ind_C, init_dens_triple)
 
dens_f = well3_finite.density()
[q_f_f, q_b_f] = well3_finite.committor()
 
# reactive density
reac_dens_f = well3_finite.reac_density()

# and reactive currents
[current_f, eff_current_f] = well3_finite.reac_current()

rate_f = well3_finite.transition_rate()



###########################################################################
#plotting
def plot_subplot_3well(data, datashape, extent, timeframe, size, v_min, v_max, title, subtitles=None):
    fig, ax = plt.subplots(1, timeframe, sharex='col',
                           sharey='row', figsize=size)
    if timeframe == 1:
        plt.imshow(data.reshape(datashape), ax=ax, vmin=v_min, vmax=v_max, origin='lower', extent=extent)
    else:
        for i in range(timeframe):
            ax[i].imshow(data[i,:].reshape(datashape), vmin=v_min, vmax=v_max, origin='lower', extent=extent)
            if subtitles is not None:
                ax[i].set_title(subtitles[i])  # , pad=0)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    return fig


#############################################################################
# plots  infinite-time, ergodic
plt.figure(1)
plt.imshow(stat_dens.reshape((xdim,ydim)), origin='lower', extent = (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) )
plt.figure(2)
plt.imshow(reac_dens.reshape((xdim,ydim)), origin='lower', extent = (interval[0,0],interval[0,1],interval[1,0],interval[1,1]))
plt.figure(3)
plt.imshow(q_f.reshape((xdim,ydim)), origin='lower', extent = (interval[0,0],interval[0,1],interval[1,0],interval[1,1]))
plt.figure(4)
plt.imshow(q_b.reshape((xdim,ydim)), origin='lower', extent = (interval[0,0],interval[0,1],interval[1,0],interval[1,1]))


#calculation the effective vector for each state
eff_vectors = np.zeros((dim_st, 2))
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  
        eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  

plt.figure(5)
plt.quiver(xn,yn,list(eff_vectors[:,0]),list(eff_vectors[:,1]))

######################################################## plots periodic

subtitles_p = np.array(['m = ' + str(i) for i in np.arange(M)])

data = stat_dens_p
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, 'Periodic stationary density', subtitles = subtitles_p)
fig.savefig(os.path.join(charts_path, 'triplewell_dens_p.png'), dpi=100)


data = q_f_p
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, 'Periodic forward committor', subtitles = subtitles_p)
fig.savefig(os.path.join(charts_path, 'triplewell_q_f_p.png'), dpi=100)

data = q_b_p
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, 'Periodic backward committor', subtitles = subtitles_p)
fig.savefig(os.path.join(charts_path, 'triplewell_q_p_p.png'), dpi=100)

data = reac_dens_p
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, 'Reactive periodic density', subtitles = subtitles_p)
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_p.png'), dpi=100)


######################################################## plots finite-time

subtitles_f = np.array(['n = ' + str(i) for i in np.arange(N)])

data = dens_f
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, 'Finite-time stationary density', subtitles = subtitles_f)
fig.savefig(os.path.join(charts_path, 'triplewell_dens_f.png'), dpi=100)


data = q_f_f
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, 'Finite-time forward committor', subtitles = subtitles_f)
fig.savefig(os.path.join(charts_path, 'triplewell_q_f_f.png'), dpi=100)

data = q_b_f
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, 'Finite-time backward committor', subtitles = subtitles_f)
fig.savefig(os.path.join(charts_path, 'triplewell_q_p_f.png'), dpi=100)

data = reac_dens_f
v_min = np.min(data)
v_max = np.max(data)
fig = plot_subplot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, 'Finite-time periodic density', subtitles = subtitles_f)
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f.png'), dpi=100)
