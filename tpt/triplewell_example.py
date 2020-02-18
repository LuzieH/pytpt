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
charts_path = os.path.join(my_path, 'charts')
T = np.load(os.path.join(my_path, 'data/triplewell_T.npy'))
T_m = np.load(os.path.join(my_path, 'data/triplewell_T_m.npy'))
T_small_noise = np.load(os.path.join(my_path, 'data/triplewell_T_small_noise.npy'))
#T_m_small_noise = np.load(os.path.join(my_path, 'data/triplewell_T_m_small_noise.npy'))
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

#define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A]=1
densAB[ind_B]=1

example_name = 'triplewell'
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

N_bif_array = np.array([20, 50, 100, 500])#time window 20-> lower channel only in stat dens, time window 50, lower channel in both
N_bif_size = np.shape(N_bif_array)[0]

norm_reac_dens_f_bif_all = np.zeros((N_bif_size,dim_st))
eff_current_f_bif_all = np.zeros((N_bif_size,dim_st,2))  
color_current_f_bif_all = np.zeros((N_bif_size,dim_st))

subtitles_bif_dens = []
subtitles_bif_eff = []

ind=0
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
            if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
                eff_vectors_f_bif[i,0] += eff_current_f_bif[int(N_bif/2),i,j] *  (xn[j] - xn[i]) *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_f_bif[i,1] += eff_current_f_bif[int(N_bif/2),i,j] *  (yn[j] - yn[i]) *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]])))  
        colors_f_bif[i] = np.linalg.norm(eff_vectors_f_bif[i,:])
        if colors_f_bif[i]>0:
            eff_vectors_unit_f_bif[i,:] = eff_vectors_f_bif[i,:]/colors_f_bif[i] 
            
            
    eff_current_f_bif_all[ind,:,:] = eff_vectors_unit_f_bif
    color_current_f_bif_all[ind,:] = colors_f_bif
    
    subtitles_bif_dens.append('$\hat{\mu}^{AB}$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
    subtitles_bif_eff.append('$f^+$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
    
    ind=ind+1
    
###############################################################
#plots bifurcation analysis, small noise

 
 
data = norm_reac_dens_f_bif_all
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N_bif_size, (3*N_bif_size,3), v_min, v_max,subtitles_bif_dens,background=densAB)
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f_bif.png'), dpi=100,bbox_inches='tight')

            
fig = plot_3well_effcurrent(eff_current_f_bif_all,color_current_f_bif_all, xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), N_bif_size, (3*N_bif_size,3),subtitles_bif_eff )
fig.savefig(os.path.join(charts_path, 'triplewell_eff_f_bif.png'), dpi=100,bbox_inches='tight')


############################################################################
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
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\hat{\mu}^{AB}$'],background=densAB)
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
        if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
            eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
            eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
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
fig = plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , M, (3*M,3), v_min, v_max,np.array(['$\hat{\mu}^{AB}_0$','$\hat{\mu}^{AB}_1$','$\hat{\mu}^{AB}_2$','$\hat{\mu}^{AB}_3$','$\hat{\mu}^{AB}_4$','$\hat{\mu}^{AB}_5$']),background=densAB) 
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
            if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
                eff_vectors_p[m,i,0] += eff_current_p[m,i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_p[m,i,1] += eff_current_p[m,i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
        colors_p[m,i] = np.linalg.norm(eff_vectors_p[m,i,:])
        if colors_p[m,i]>0:
            eff_vectors_unit_p[m,i,:] = eff_vectors_p[m,i,:]/colors_p[m,i]
 


 
            
fig = plot_3well_effcurrent(eff_vectors_unit_p, colors_p, xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), M, (3*M,3), subtitles_m('$f^+_{}$',M)) 
fig.savefig(os.path.join(charts_path, 'triplewell_eff_p.png'), dpi=100,bbox_inches='tight')


plot_rate(
    rate=rate_p,
    time_av_rate=time_av_rate_p,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_p.png'),
    title='',xlabel = 'm', average_rate_legend=r'$ \bar{k}^{AB}_M $'
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
fig = plot_3well(data[1:N-1,:], (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , N-2, (3*(N-2),3), v_min, v_max, np.array(['$\hat{\mu}^{AB}(1)$','$\hat{\mu}^{AB}(2)$','$\hat{\mu}^{AB}(3)$','$\hat{\mu}^{AB}(4)$']),background=densAB) 
fig.savefig(os.path.join(charts_path, 'triplewell_reac_dens_f.png'), dpi=100,bbox_inches='tight')

#calculation the effective vector for each state
eff_vectors_f = np.zeros((N,dim_st, 2))
eff_vectors_unit_f = np.zeros((N,dim_st, 2))
colors_f = np.zeros((N,dim_st))
for n in np.arange(N):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            #if np.isnan(eff_current_f[n,i,j])==False:

            if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
                eff_vectors_f[n,i,0] += eff_current_f[n,i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_f[n,i,1] += eff_current_f[n,i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
        colors_f[n,i] = np.linalg.norm(eff_vectors_f[n,i,:])
        if colors_f[n,i]>0:
            eff_vectors_unit_f[n,i,:] = eff_vectors_f[n,i,:]/colors_f[n,i]
            

fig = plot_3well_effcurrent(eff_vectors_unit_f[:N-1,:,:], colors_f[:N-1,:], xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), N-1, (3*(N-1),3), subtitles_m('$f^+({})$',N-1))
fig.savefig(os.path.join(charts_path, 'triplewell_eff_f.png'), dpi=100,bbox_inches='tight')

plot_rate(
    rate=rate_f,
    time_av_rate=time_av_rate_f,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_f.png'),
    title='Discrete finite-time, time-homogeneous rates',xlabel = 'n', average_rate_legend=r'$\bar{k}^{AB}_N$'
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_f.png'),
    title='Discrete finite-time, time-homogeneous reactiveness',
)


 
