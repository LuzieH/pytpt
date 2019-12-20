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
 
##############################################################################
# load data
 
my_path = os.path.abspath(os.path.dirname(__file__))
T = np.load(os.path.join(my_path, 'data/triplewell_T.npy'))
T_m = np.load(os.path.join(my_path, 'data/triplewell_T_m.npy'))
T_small_noise = np.load(os.path.join(my_path, 'data/triplewell_T_small_noise.npy'))
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

# therof compute the normalized reactive density
norm_reac_dens = well3.norm_reac_density()

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

# normalized reactive density
reac_norm_factor_p = well3_periodic.reac_norm_factor()
norm_reac_dens_p = well3_periodic.norm_reac_density()

# and reactive currents
[current_p, eff_current_p] = well3_periodic.reac_current()
 
[rate_p, time_av_rate_p] = well3_periodic.transition_rate()

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

###################################
# finite time bifurcation analysis 

#finite-time

def Tn_small_noise(n):  
    return T_small_noise#T_m[np.mod(m,M),:,:].squeeze()

# compute stationary density of triple well with small noise to get initial density
well3_small_noise = tp.transitions_mcs(T_small_noise, ind_A, ind_B, ind_C)
stat_dens_small_noise = well3_small_noise.stationary_density()

init_dens_triple_bif = stat_dens_small_noise

N_bif_array = np.array([20,30, 40,50, 60, 100, 150, 250, 500])#time window 20-> lower channel only in stat dens, time window 50, lower channel in both
    
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
    
    ###############################################################
    #plots bifurcation analysis, small noise
    
    
    data = np.array([dens_f_bif[int(N_bif/2)]])
    v_min = np.nanmin(data)
    v_max = np.nanmax(data)
    fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\pi$('+str(int(N_bif/2))+'), $N=$'+str(N_bif)])
    fig.savefig(os.path.join(charts_path, 'triplewell_dens_f_bif'+str(N_bif)+'.png'), dpi=100,bbox_inches='tight')
    
    
    data = np.array([q_f_f_bif[int(N_bif/2)]])
    v_min = np.nanmin(data)
    v_max = np.nanmax(data)
    fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$q^+$('+str(int(N_bif/2))+'), $N=$'+str(N_bif)])
    fig.savefig(os.path.join(charts_path, 'triplewell_q_f_f_bif'+str(N_bif)+'.png'), dpi=100,bbox_inches='tight')
    
    data = np.array([q_b_f_bif[int(N_bif/2)]])
    v_min = np.nanmin(data)
    v_max = np.nanmax(data)
    fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$q^-$('+str(int(N_bif/2))+'), $N=$'+str(N_bif)])
    fig.savefig(os.path.join(charts_path, 'triplewell_q_b_f_bif'+str(N_bif)+'.png'), dpi=100,bbox_inches='tight')
    
    data = np.array([norm_reac_dens_f_bif[int(N_bif/2)]])
    v_min = np.nanmin(data)
    v_max = np.nanmax(data)
    fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\mu^\mathcal{AB}$('+str(int(N_bif/2))+'), $N=$'+str(N_bif)])
    fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f_bif'+str(N_bif)+'.png'), dpi=100,bbox_inches='tight')
    
    #define AB sets
    densAB = np.zeros(dim_st)
    densAB[ind_A]=1
    densAB[ind_B]=1
    
    #calculation the effective vector for each state
    eff_vectors_f_bif = np.zeros((dim_st, 2))
    eff_vectors_unit_f_bif = np.zeros((dim_st, 2))
    colors_f_bif = np.zeros(dim_st)
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            eff_vectors_f_bif[i,0] += eff_current_f_bif[int(N_bif/2),i,j] *  (xn[j] - xn[i])  
            eff_vectors_f_bif[i,1] += eff_current_f_bif[int(N_bif/2),i,j] *  (yn[j] - yn[i])  
        colors_f_bif[i] = np.linalg.norm(eff_vectors_f_bif[i,:])
        if colors_f_bif[i]>0:
            eff_vectors_unit_f_bif[i,:] = eff_vectors_f_bif[i,:]/colors_f_bif[i] 
                
    fig = plot_3well_effcurrent(np.array([eff_vectors_unit_f_bif]), np.array([colors_f_bif]), xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), 1, (3*1,3),['$f^+$('+str(int(N_bif/2))+'), $N=$'+str(N_bif)])
    fig.savefig(os.path.join(charts_path, 'triplewell_eff_f_bif'+str(N_bif)+'.png'), dpi=100,bbox_inches='tight')
    


########################################
#finite-time, periodic forcing

N_force = 6 #time window

# initial density
init_dens_triple_force = stat_dens_p[0, :]

# instantiate
well3_finite_force = tpf.transitions_finite_time(Tm, N_force, ind_A, ind_B,  ind_C, init_dens_triple_force)
 
dens_f_force = well3_finite_force.density()
[q_f_f_force, q_b_f_force] = well3_finite_force.committor()
 
# normalized reactive density
reac_norm_factor_f_force = well3_finite_force.reac_norm_factor()
norm_reac_dens_f_force = well3_finite_force.norm_reac_density()

# and reactive currents
[current_f_force, eff_current_f_force] = well3_finite_force.reac_current()

[rate_f_force, time_av_rate_f_force] = well3_finite_force.transition_rate()

charts_path = os.path.join(my_path, 'charts')
example_name = 'triplewell'
#############################https://de.overleaf.com/project/5d555a6bdfb42d0001a9dac6################################################
## plots  infinite-time, ergodic


data = np.array([stat_dens])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\pi$'])
fig.savefig(os.path.join(charts_path, 'triplewell_dens.png'), dpi=100,bbox_inches='tight')


data = np.array([q_f])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$q^+$'])
fig.savefig(os.path.join(charts_path, 'triplewell_q_f.png'), dpi=100,bbox_inches='tight')

data = np.array([q_b])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$q^-$'])
fig.savefig(os.path.join(charts_path, 'triplewell_q_b.png'), dpi=100,bbox_inches='tight')

data = np.array([norm_reac_dens])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\mu^\mathcal{AB}$'])
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens.png'), dpi=100,bbox_inches='tight')

#define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A]=1
densAB[ind_B]=1

#calculation the effective vector for each state
eff_vectors = np.zeros((dim_st, 2))
eff_vectors_unit = np.zeros((dim_st, 2))
colors = np.zeros(dim_st)
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  
        eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  
    colors[i] = np.linalg.norm(eff_vectors[i,:])
    if colors[i]>0:
        eff_vectors_unit[i,:] = eff_vectors[i,:]/colors[i] 
            
fig = plot_3well_effcurrent(np.array([eff_vectors_unit]), np.array([colors]), xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), 1, (3*1,3),['$f^+$'])
fig.savefig(os.path.join(charts_path, 'triplewell_eff.png'), dpi=100,bbox_inches='tight')



######################################################## plots periodic

def subtitles_m(quant,M):
    return np.array([quant.format(str(i)) for i in np.arange(M)])

data = stat_dens_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, subtitles_m('$\pi_{}$',M))#Periodic stationary density', subtitles = subtitles_p)
fig.savefig(os.path.join(charts_path, 'triplewell_dens_p.png'), dpi=100,bbox_inches='tight')


data = q_f_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, subtitles_m('$q^+_{}$',M))
fig.savefig(os.path.join(charts_path, 'triplewell_q_f_p.png'), dpi=100,bbox_inches='tight')

data = q_b_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max, subtitles_m('$q^-_{}$',M))
fig.savefig(os.path.join(charts_path, 'triplewell_q_b_p.png'), dpi=100,bbox_inches='tight')

data = norm_reac_dens_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max,np.array(['$\mu^\mathcal{AB}_0$','$\mu^\mathcal{AB}_1$','$\mu^\mathcal{AB}_2$','$\mu^\mathcal{AB}_3$','$\mu^\mathcal{AB}_4$','$\mu^\mathcal{AB}_5$'])) 
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_p.png'), dpi=100,bbox_inches='tight')

#define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A]=1
densAB[ind_B]=1

#calculation the effective vector for each state
eff_vectors_p = np.zeros((M,dim_st, 2))
eff_vectors_unit_p = np.zeros((M,dim_st, 2))
colors_p = np.zeros((M,dim_st))
for m in np.arange(M):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            eff_vectors_p[m,i,0] += eff_current_p[m,i,j] *  (xn[j] - xn[i])  
            eff_vectors_p[m,i,1] += eff_current_p[m,i,j] *  (yn[j] - yn[i])  
        colors_p[m,i] = np.linalg.norm(eff_vectors_p[m,i,:])
        if colors_p[m,i]>0:
            eff_vectors_unit_p[m,i,:] = eff_vectors_p[m,i,:]/colors_p[m,i]
            
fig = plot_3well_effcurrent(eff_vectors_unit_p, colors_p, xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), M, (3*M,3), subtitles_m('$f^+_{}$',M)) 
fig.savefig(os.path.join(charts_path, 'triplewell_eff_p.png'), dpi=100,bbox_inches='tight')


plot_rate(
    rate=rate_p,
    time_av_rate=time_av_rate_p,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_p.png'),
    title='Discrete periodic rates',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_p,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_p.png'),
    title='Discrete periodic reactiveness',
)
######################################################## plots finite-time


data = dens_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, subtitles_m('$\lambda({})$',N))
fig.savefig(os.path.join(charts_path, 'triplewell_dens_f.png'), dpi=100,bbox_inches='tight')


data = q_f_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, subtitles_m('$q^+({})$',N))
fig.savefig(os.path.join(charts_path, 'triplewell_q_f_f.png'), dpi=100,bbox_inches='tight')

data = q_b_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N, (3*N,3), v_min, v_max, subtitles_m('$q^-({})$',N))
fig.savefig(os.path.join(charts_path, 'triplewell_q_b_f.png'), dpi=100,bbox_inches='tight')

data = norm_reac_dens_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data[1:N-1,:], (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N-2, (3*(N-2),3), v_min, v_max, np.array(['$\mu^\mathcal{AB}(1)$','$\mu^\mathcal{AB}(2)$','$\mu^\mathcal{AB}(3)$','$\mu^\mathcal{AB}(4)$'])) 
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f.png'), dpi=100,bbox_inches='tight')

#calculation the effective vector for each state
eff_vectors_f = np.zeros((N,dim_st, 2))
eff_vectors_unit_f = np.zeros((N,dim_st, 2))
colors_f = np.zeros((N,dim_st))
for n in np.arange(N):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            #if np.isnan(eff_current_f[n,i,j])==False:
            eff_vectors_f[n,i,0] += eff_current_f[n,i,j] *  (xn[j] - xn[i])  
            eff_vectors_f[n,i,1] += eff_current_f[n,i,j] *  (yn[j] - yn[i])  
        colors_f[n,i] = np.linalg.norm(eff_vectors_f[n,i,:])
        if colors_f[n,i]>0:
            eff_vectors_unit_f[n,i,:] = eff_vectors_f[n,i,:]/colors_f[n,i]
            

fig = plot_3well_effcurrent(eff_vectors_unit_f[:N-1,:,:], colors_f[:N-1,:], xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), N-1, (3*(N-1),3), subtitles_m('$f^+({})$',N-1))
fig.savefig(os.path.join(charts_path, 'triplewell_eff_f.png'), dpi=100,bbox_inches='tight')

plot_rate(
    rate=rate_f,
    time_av_rate=time_av_rate_f,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_f.png'),
    title='Discrete finite-time, time-homogeneous rates',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_f.png'),
    title='Discrete finite-time, time-homogeneous reactiveness',
)

######################################################## plots finite-time, forcing

data = dens_f_force
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N_force, (3*N_force,3), v_min, v_max, subtitles_m('$\lambda({})$',N_force))
fig.savefig(os.path.join(charts_path, 'triplewell_dens_f_force.png'), dpi=100,bbox_inches='tight')


data = q_f_f_force
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N_force, (3*N_force,3), v_min, v_max, subtitles_m('$q^+({})$',N_force))
fig.savefig(os.path.join(charts_path, 'triplewell_q_f_f_force.png'), dpi=100,bbox_inches='tight')

data = q_b_f_force
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N_force, (3*N_force,3), v_min, v_max, subtitles_m('$q^-({})$',N_force))
fig.savefig(os.path.join(charts_path, 'triplewell_q_b_f_force.png'), dpi=100,bbox_inches='tight')


data = norm_reac_dens_f_force
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data[1:N_force-1,:], (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N_force-2, (3*(N_force-2),3), v_min, v_max, np.array(['$\mu^\mathcal{AB}(1)$','$\mu^\mathcal{AB}(2)$','$\mu^\mathcal{AB}(3)$','$\mu^\mathcal{AB}(4)$'])) 
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f_force.png'), dpi=100,bbox_inches='tight')

#calculation the effective vector for each state
eff_vectors_f_force = np.zeros((N_force, dim_st, 2))
eff_vectors_unit_f_force = np.zeros((N_force, dim_st, 2))
colors_f_force = np.zeros((N_force, dim_st))
for n in np.arange(N_force):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            #if np.isnan(eff_current_f[n,i,j])==False:
            eff_vectors_f_force[n,i,0] += eff_current_f_force[n,i,j] *  (xn[j] - xn[i])  
            eff_vectors_f_force[n,i,1] += eff_current_f_force[n,i,j] *  (yn[j] - yn[i])  
        colors_f_force[n,i] = np.linalg.norm(eff_vectors_f_force[n,i,:])
        if colors_f_force[n,i]>0:
            eff_vectors_unit_f_force[n,i,:] = eff_vectors_f_force[n,i,:]/colors_f_force[n,i]
            

fig = plot_3well_effcurrent(eff_vectors_unit_f_force[:N_force-1,:,:], colors_f_force[:N_force-1,:], xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), N_force-1, (3*(N_force-1),3), subtitles_m('$f^+({})$',N_force-1))
fig.savefig(os.path.join(charts_path, 'triplewell_eff_f_force.png'), dpi=100,bbox_inches='tight')

plot_rate(
    rate=rate_f_force,
    time_av_rate=time_av_rate_f_force,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_f_force.png'),
    title='Discrete finite-time, periodic forcing rates',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f_force,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_f_force.png'),
    title='Discrete finite-time, periodic forcing reactiveness',
)

