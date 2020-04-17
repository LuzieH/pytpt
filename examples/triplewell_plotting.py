from plotting import plot_3well, \
                     plot_3well_effcurrent, \
                     plot_rate, \
                     plot_reactiveness
import numpy as np
 

import os.path

# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
charts_path = os.path.join(my_path, 'charts')

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
# T_small_noise = triplewell_construction['T_small_noise']


# load triple well statistics data 
example_name = 'triplewell_stationary'
triplewell_ergodic = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'triplewell_periodic'
triplewell_periodic = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'triplewell_finite'
triplewell_finite = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'triplewell_bifurcation'
triplewell_bif = np.load(
    os.path.join(data_path, example_name + '.npz')
)

stat_dens = triplewell_ergodic['stat_dens']
q_f = triplewell_ergodic['q_f']
q_b = triplewell_ergodic['q_b']
reac_norm_factor = triplewell_ergodic['reac_norm_factor']
norm_reac_dens = triplewell_ergodic['norm_reac_dens']
eff_current = triplewell_ergodic['eff_current']
rate = triplewell_ergodic['rate']
length = triplewell_ergodic['length']

stat_dens_p = triplewell_periodic['stat_dens']
q_f_p = triplewell_periodic['q_f']
q_b_p = triplewell_periodic['q_b']
reac_norm_factor_p = triplewell_periodic['reac_norm_factor']
norm_reac_dens_p = triplewell_periodic['norm_reac_dens']
eff_current_p = triplewell_periodic['eff_current']
rate_p = triplewell_periodic['rate']
time_av_rate_p = triplewell_periodic['time_av_rate']
av_length_p = triplewell_periodic['av_length']

dens_f = triplewell_finite['dens']
q_f_f = triplewell_finite['q_f']
q_b_f = triplewell_finite['q_b']
reac_norm_factor_f = triplewell_finite['reac_norm_factor']
norm_reac_dens_f = triplewell_finite['norm_reac_dens']
eff_current_f = triplewell_finite['eff_current']
rate_f = triplewell_finite['rate']
time_av_rate_f = triplewell_finite['time_av_rate']
av_length_f = triplewell_finite['av_length']

norm_reac_dens_f_bif_all = triplewell_bif['norm_reac_dens']
eff_current_f_bif_all = triplewell_bif['eff_current']
color_current_f_bif_all = triplewell_bif['color_current']


# state space
x = np.arange(interval[0,0], interval[0,1] + dx, dx) # box centers in x and y direction
y = np.arange(interval[1,0] ,interval[1,1] + dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] # discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim * ydim # dimension of the statespace
xn = np.reshape(xv,(xdim * ydim,1))
yn = np.reshape(yv,(xdim * ydim,1))
grid = np.squeeze(np.array([xn,yn]))

# define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A] = 1
densAB[ind_B] = 1

print("rate (infinite-time, stationary): %f" % rate)
print("periodic-averaged rate (infinite-time, periodic): %f" % time_av_rate_p[0])
print("time-averaged rate (finite-time, time-homogeneous): %f" % time_av_rate_f[0])

print("mean length (infinite-time, stationary): %f" % length)
print("mean length (infinite-time, periodic): %f" % av_length_p)
print("mean length (finite-time, time-homogeneous): %f" % av_length_f)


# plots infinite-time, ergodic
example_name = 'triplewell_stationary'

data = np.array([stat_dens])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'dens.png')
plot_3well(data, (xdim, ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), 1, (3*1,3), v_min, v_max, ['$\pi$'], file_path)

data = np.array([q_f])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_f.png')
plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1], interval[1,0], interval[1,1]), 1, (3*1,3), v_min, v_max, ['$q^+$'], file_path)

data = np.array([q_b])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_b.png')
plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), 1, (3*1,3), v_min, v_max, ['$q^-$'], file_path)

data = np.array([norm_reac_dens])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'reac_dens.png')
plot_3well(data, (xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]) , 1, (3*1,3), v_min, v_max, ['$\hat{\mu}^{AB}$'], file_path, background=densAB)

# define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A] = 1
densAB[ind_B] = 1

# calculation the effective vector for each state
eff_vectors = np.zeros((dim_st, 2))
eff_vectors_unit = np.zeros((dim_st, 2))
colors = np.zeros(dim_st)
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        if np.linalg.norm(np.array([xn[j] - xn[i], yn[j] - yn[i]])) > 0:
            eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
            eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
    colors[i] = np.linalg.norm(eff_vectors[i,:])
    if colors[i] > 0:
        eff_vectors_unit[i,:] = eff_vectors[i,:]/colors[i] 
            
file_path = os.path.join(charts_path, example_name + '_' + 'eff.png')
plot_3well_effcurrent(np.array([eff_vectors_unit]), np.array([colors]), xn, yn, densAB, (xdim, ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), 1, (3*1, 3), ['$f^+$'], file_path)

# plots periodic
example_name = 'triplewell_periodic'
M = np.shape(T_m)[0]

def subtitles_m(quant,M):
    return np.array([quant.format(str(i)) for i in np.arange(M)])

data = stat_dens_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'dens.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), M, (3*M,3), v_min, v_max, subtitles_m('$\pi_{}$', M), file_path)#Periodic stationary density', subtitles = subtitles_p)

data = q_f_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_f.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), M, (3*M,3), v_min, v_max, subtitles_m('$q^+_{}$', M), file_path)

data = q_b_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_b.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), M, (3*M,3), v_min, v_max, subtitles_m('$q^-_{}$', M), file_path)

data = norm_reac_dens_p
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'reac_dens.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), M, (3*M,3), v_min, v_max,np.array(['$\hat{\mu}^{AB}_0$','$\hat{\mu}^{AB}_1$','$\hat{\mu}^{AB}_2$','$\hat{\mu}^{AB}_3$','$\hat{\mu}^{AB}_4$','$\hat{\mu}^{AB}_5$']), file_path, background=densAB) 

# define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A] = 1
densAB[ind_B] = 1

# calculation the effective vector for each state
eff_vectors_p = np.zeros((M, dim_st, 2))
eff_vectors_unit_p = np.zeros((M, dim_st, 2))
colors_p = np.zeros((M, dim_st))
for m in np.arange(M):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            if np.linalg.norm(np.array([xn[j] - xn[i], yn[j] - yn[i]])) > 0:
                eff_vectors_p[m,i,0] += eff_current_p[m,i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_p[m,i,1] += eff_current_p[m,i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
        colors_p[m,i] = np.linalg.norm(eff_vectors_p[m,i,:])
        if colors_p[m,i]>0:
            eff_vectors_unit_p[m,i,:] = eff_vectors_p[m,i,:]/colors_p[m,i]
 
file_path = os.path.join(charts_path, example_name + '_' + 'eff.png')
plot_3well_effcurrent(eff_vectors_unit_p, colors_p, xn, yn, densAB, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), M, (3*M,3), subtitles_m('$f^+_{}$', M), file_path) 

plot_rate(
    rate=rate_p,
    time_av_rate=time_av_rate_p,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates.png'),
    title='',
    xlabel = 'm',
    average_rate_legend=r'$ \bar{k}^{AB}_M $',
)
 
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_p,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness.png'),
    title='Discrete periodic reactiveness',
)
exit()

# plots finite-time
example_name = 'triplewell_finite'
N = 6 # time window

data = dens_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'dens.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N, (3*N,3), v_min, v_max, subtitles_m('$\lambda({})$',N), file_path)

data = q_f_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_f.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N, (3*N,3), v_min, v_max, subtitles_m('$q^+({})$',N), file_path)

data = q_b_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'q_b.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N, (3*N,3), v_min, v_max, subtitles_m('$q^-({})$',N), file_path)

data = norm_reac_dens_f
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'dens.png')
plot_3well(data[1:N-1,:], (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N-2, (3*(N-2),3), v_min, v_max, np.array(['$\hat{\mu}^{AB}(1)$','$\hat{\mu}^{AB}(2)$','$\hat{\mu}^{AB}(3)$','$\hat{\mu}^{AB}(4)$']), file_path, background=densAB) 

# calculation the effective vector for each state
eff_vectors_f = np.zeros((N,dim_st, 2))
eff_vectors_unit_f = np.zeros((N,dim_st, 2))
colors_f = np.zeros((N,dim_st))
for n in np.arange(N):
    for i in np.arange(dim_st):
        for j in np.arange(dim_st):
            # if np.isnan(eff_current_f[n,i,j])==False:

            if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
                eff_vectors_f[n,i,0] += eff_current_f[n,i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
                eff_vectors_f[n,i,1] += eff_current_f[n,i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
        colors_f[n,i] = np.linalg.norm(eff_vectors_f[n,i,:])
        if colors_f[n,i]>0:
            eff_vectors_unit_f[n,i,:] = eff_vectors_f[n,i,:]/colors_f[n,i]
            

file_path = os.path.join(charts_path, example_name + '_' + 'eff.png')
plot_3well_effcurrent(eff_vectors_unit_f[:N-1,:,:], colors_f[:N-1,:], xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), N-1, (3*(N-1),3), subtitles_m('$f^+({})$',N-1), file_path)

plot_rate(
    rate=rate_f,
    time_av_rate=time_av_rate_f,                                                               
    file_path=os.path.join(charts_path, example_name + '_' + 'rates.png'),
    title='Discrete finite-time, time-homogeneous rates',
    xlabel = 'n',
    average_rate_legend=r'$\bar{k}^{AB}_N$',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness.png'),
    title='Discrete finite-time, time-homogeneous reactiveness',
)

# plots bifurcation analysis, small noise
example_name = 'triplewell_bifurcation'
# time window 20-> lower channel only in stat dens, time window 50, lower channel in both
N_bif_array = np.array([20, 50, 100, 500])
N_bif_size = np.shape(N_bif_array)[0]
subtitles_bif_dens = []
subtitles_bif_eff = []
for N_bif in N_bif_array:
    subtitles_bif_dens.append('$\hat{\mu}^{AB}$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
    subtitles_bif_eff.append('$f^+$('+str(int(N_bif/2))+'), $N=$'+str(N_bif))
 
data = norm_reac_dens_f_bif_all
v_min = np.nanmin(data)
v_max = np.nanmax(data)
file_path = os.path.join(charts_path, example_name + '_' + 'reac_dens.png')
plot_3well(data, (xdim,ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N_bif_size, (3*N_bif_size,3), v_min, v_max, subtitles_bif_dens, file_path, background=densAB)

file_path = os.path.join(charts_path, example_name + '_' + 'eff.png')
plot_3well_effcurrent(eff_current_f_bif_all,color_current_f_bif_all, xn, yn, densAB, (xdim, ydim), (interval[0,0], interval[0,1], interval[1,0], interval[1,1]), N_bif_size, (3*N_bif_size, 3), subtitles_bif_eff, file_path)
