import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

from plotting import plot_network_density as plot_density, \
                     plot_network_effective_current as plot_effective_current, \
                     plot_network_effcurrent_and_rate as plot_effcurrent_and_rate, \
                     plot_rate, \
                     plot_reactiveness, \
                     plot_convergence, \
                     plot_colorbar_only

import numpy as np
import networkx as nx

import os.path

# TODO add colorbar to plots

# general

# load data about small network
my_path = os.path.abspath(os.path.dirname(__file__))
states = np.load(
    os.path.join(my_path, 'data/small_network_states.npy'),
    allow_pickle=True, 
)
states = states.item()
labels = np.load(
    os.path.join(my_path, 'data/small_network_labels.npy'),
    allow_pickle=True, 
)
labels = labels.item()
pos = np.load(
    os.path.join(my_path, 'data/small_network_pos.npy'),
    allow_pickle=True,
)
pos = pos.item()
T = np.load(os.path.join(my_path, 'data/small_network_T.npy'))
L = np.load(os.path.join(my_path, 'data/small_network_L.npy'))
K = np.load(os.path.join(my_path, 'data/small_network_K.npy'))

S = len(states)

ind_A = np.array([0])
ind_C = np.arange(1, np.shape(T)[0] - 1)
ind_B = np.array([4])


# TPT ergodic, infinite-time

# transition matrix
P = T + L

# instantiate
small = tp.transitions_mcs(P, ind_A, ind_B, ind_C)
stat_dens = small.stationary_density()

# compute committor probabilities
[q_f, q_b] = small.committor()

# therof compute the normalized reactive density
norm_reac_dens = small.norm_reac_density()

# and reactive currents
[current, eff_current] = small.reac_current()
rate = small.transition_rate()  # AB discrete transition rate

mean_length = small.mean_transition_length()

#compute share along upper (1) and lower path (via 3)
eff_out = eff_current[0,1]+eff_current[0,3]
share_1 = eff_current[0,1]/eff_out
share_3 = eff_current[0,3]/eff_out
print('In the infinite-time, stationary case, a share of '+str(share_3)+' outflow is via 3, while a share of '+str(share_1)+' outflow is via 1')


# TPT periodisch
# use as transition matrix T + wL, where w varies from 1..0..-1...0
# either faster switching or slower dynamics

M = 6  # 6 size of period

# transition matrix at time k
def P_p(k):
    # varies the transition matrices periodically, by weighting the added
    # matrix L with weights 1..0..-1.. over one period
    return T + np.cos(k*2.*np.pi/M)*L


# instantiate
small_periodic = tpp.transitions_periodic(P_p, M, ind_A, ind_B, ind_C)
stat_dens_p = small_periodic.stationary_density()

[q_f_p, q_b_p] = small_periodic.committor()
P_back_m = small_periodic.backward_transitions()

# normalized reactive density
norm_reac_dens_p = small_periodic.norm_reac_density()

# and reactive currents
[current_p, eff_current_p] = small_periodic.reac_current()

[rate_p, time_av_rate_p] = small_periodic.transition_rate()

mean_length_p = small_periodic.mean_transition_length()


# TPT finite time, time-homogeneous

# transition matrix at time n
def P_hom(n):
    return P

# initial density
init_dens_small = stat_dens
N = 5  # size of time interval

# instantiate
small_finite = tpf.transitions_finite_time(
    P_hom, N, ind_A, ind_B,  ind_C, init_dens_small)
[q_f_f, q_b_f] = small_finite.committor()

stat_dens_f = small_finite.density()

# reactive density (zero at time 0 and N)
reac_norm_factor_f = small_finite.reac_norm_factor()
norm_reac_dens_f = small_finite.norm_reac_density()

# and reactive currents
[current_f, eff_current_f] = small_finite.reac_current()

# first row, out rate of A, second row in rate for B
[rate_f, time_av_rate_f] = small_finite.transition_rate()

mean_length_f = small_finite.mean_transition_length()


# TPT finite time, time-inhomogeneous
# size of time interval
N_inhom = 5 

# transition matrix at time n

def P_inhom(n):
    if np.mod(n,2)==0:
        return P + K
    else: 
        return P - K

def P_inhom_2(n):
    if n in [0, 1, 2, 7, 8, 9]: 
        return P - K/3
    elif n in [3, 6]:
        return P
    else:
        return P + K

def P_inhom_3(n):
    return np.sin(n*2.*np.pi/N_inhom)*K

# initial density
init_dens_small_inhom = stat_dens

# instantiate
small_inhom = tpf.transitions_finite_time(
    P_inhom,
    N_inhom,
    ind_A,
    ind_B,
    ind_C,
    init_dens_small_inhom,
)
[q_f_inhom, q_b_inhom] = small_inhom.committor()

stat_dens_inhom = small_inhom.density()
# reactive density (zero at time 0 and N)
reac_norm_factor_inhom = small_inhom.reac_norm_factor()
norm_reac_dens_inhom = small_inhom.norm_reac_density()

# and reactive currents
[current_inhom, eff_current_inhom] = small_inhom.reac_current()

# first row, out rate of A, second row in rate for B
[rate_inhom, time_av_rate_inhom] = small_inhom.transition_rate()

mean_length_inhom = small_inhom.mean_transition_length()

print("rate (infinite-time, stationary): %f" % rate)
print("periodic-averaged rate (infinite-time, periodic): %f" % time_av_rate_p[0])
print("time-averaged rate (finite-time, time-homogeneous): %f" % time_av_rate_f[0])
print("time-averaged rate (finite-time, time-inhomogeneous): %f" % time_av_rate_inhom[0])

print("mean length (infinite-time, stationary): %f" % mean_length)
print("mean length (infinite-time, periodic): %f" % mean_length_p)
print("mean length (finite-time, time-homogeneous): %f" % mean_length_f)
print("mean length (finite-time, time-inhomogeneous): %f" % mean_length_inhom)

# TPT finite time extension to infinite time, convergence analysis
N_max = 150  # max value of N
q_f_conv = np.zeros((N_max, S))
q_b_conv = np.zeros((N_max, S))

for n in np.arange(1, N_max + 1):
    # extended time interval
    N_ex = n*2 + 1

    # instantiate
    small_finite_ex = tpf.transitions_finite_time(
        P_hom,
        N_ex,
        ind_A,
        ind_B,
        ind_C,
        init_dens_small,
    )
    
    # compute statistics
    [q_f_ex, q_b_ex] = small_finite_ex.committor()
    q_f_conv[n-1, :] = q_f_ex[n, :]
    q_b_conv[n-1, :] = q_b_ex[n, :]


# TODO store the transition statistics in data
# idea1: 
#np.save(os.path.join(my_path, 'data/small_network_trans_stat.npy'), small)
#np.save(os.path.join(my_path, 'data/small_network_trans_stat_p.npy'), small_periodic)
#np.save(os.path.join(my_path, 'data/small_network_trans_stat_f.npy'), small_finite)
#np.save(os.path.join(my_path, 'data/small_network_trans_stat_inhom.npy'), small_inhom)

# idea2:
#C = 5
#data_coll = np.zeros((5,np.shape(stat_dens)[0]))
#data_coll[0,:] = stat_dens
#data_coll[1,:] = q_f
#data_coll[2,:] = q_b
#data_coll[3,:] = reac_dens
#subtitles_coll = np.array(['Stationary density','$q^+$','$q^-$','Reactive density','Current density'])
#
#fig = plot_subplot(data_coll, G, pos, C, (2*C, 3),'Stationary system',subtitles_coll)


# plotting
v_min_dens = min([
    np.min(stat_dens),
    np.min(stat_dens_p),
    np.min(stat_dens_f),
    np.min(stat_dens_inhom),
])
v_max_dens = max([
    np.max(stat_dens),
    np.max(stat_dens_p),
    np.max(stat_dens_f),
    np.max(stat_dens_inhom),
])
v_min_q_f = min([
    np.min(q_f),
    np.min(q_f_p),
    np.min(q_f_f),
    np.min(q_f_inhom),
])
v_max_q_f = max([
    np.max(q_f),
    np.max(q_f_p),
    np.max(q_f_f),
    np.max(q_f_inhom),
])
v_min_q_b = min([
    np.min(q_b),
    np.min(q_b_p),
    np.min(q_b_f),
    np.min(q_b_inhom),
])
v_max_q_b = max([
    np.max(q_b),
    np.max(q_b_p),
    np.max(q_b_f),
    np.max(q_b_inhom),
])
v_min_reac_dens = min([
    np.min(norm_reac_dens),
    np.min(norm_reac_dens_p),
    np.min(norm_reac_dens_f),
    np.min(norm_reac_dens_inhom),
])
v_max_reac_dens = max([
    np.max(norm_reac_dens),
    np.max(norm_reac_dens_p),
    np.max(norm_reac_dens_f),
    np.max(norm_reac_dens_inhom),
])
v_min_eff_curr = min([
    np.min(eff_current),
    np.min(eff_current_p),
    np.min(eff_current_f),
    np.min(eff_current_inhom),
])
v_max_eff_curr = max([
    np.max(eff_current),
    np.max(eff_current_p),
    np.max(eff_current_f),
    np.max(eff_current_inhom),
])


# define directory path to save the plots
charts_path = os.path.join(my_path, 'charts')
example_name = 'small_network'

# plot relative color bar
plot_colorbar_only(
    file_path=os.path.join(charts_path, example_name + '_' + 'colorbar.png'),
)
exit()

# plotting results for ergodic, infinite-time case
graphs = [nx.Graph(P)]

plot_density(
    data=np.array([stat_dens]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'dens.png'),
    subtitles=['$\pi$']
)
plot_density(
    data=np.array([q_f]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_f.png'),
    subtitles=['$q^+$'],
)
plot_density(
    data=np.array([q_b]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_b.png'),
    subtitles=['$q^-$'],
)
plot_density(
    data=np.array([norm_reac_dens]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'reac_dens.png'),
    subtitles=['$\hat{\mu}^{AB}$'],
)
plot_effcurrent_and_rate(
    eff_current=np.array([eff_current]),
    shifted_rate=[[rate, rate]],
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(charts_path, example_name + '_' + 'eff.png'),
    subtitles=['$f^+$'],
)

# plotting results for periodic case
graphs_p = [nx.Graph(P_p(m)) for m in np.arange(M)] 
shifted_rate_p = np.zeros((M, 2))
shifted_rate_p[:, 0] = rate_p[:, 0]
shifted_rate_p[:M-1, 1] = rate_p[1:, 1]
shifted_rate_p[M-1, 1] = rate_p[0, 1]

plot_density(
    data=stat_dens_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'dens_p.png'),
    subtitles=['$\pi_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=q_f_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_f_p.png'),
    subtitles=['$q^+_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=q_b_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_b_p.png'),
    subtitles=['$q^-_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=norm_reac_dens_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'reac_dens_p.png'),
    subtitles=['$\hat{\mu}_' + str(m) + '^{AB}$' for m in np.arange(M)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_p,
    shifted_rate=shifted_rate_p,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(charts_path, example_name + '_' + 'eff_p.png'),
    subtitles=['$f^+_' + str(m) + '$' for m in np.arange(M)]
)
plot_rate(
    rate=rate_p,
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_p.png'),
    title='Discrete M-periodic rates',xlabel = 'm', average_rate_legend=r'$\bar{k}^{AB}_M$'
)


# plotting results for finite-time, time-homogeneous case
graphs_f = [nx.Graph(P_hom(n)) for n in np.arange(N)] 
shifted_rate_f = np.zeros((N-1, 2))
shifted_rate_f[:, 0] = rate_f[:N-1, 0]
shifted_rate_f[:, 1] = rate_f[1:, 1]

plot_density(
    data=stat_dens_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'dens_f.png'),
    subtitles=['$\pi(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=q_f_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_f_f.png'),
    subtitles=['$q^+(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=q_b_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_b_f.png'),
    subtitles=['$q^-(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=norm_reac_dens_f[1:N-1],
    graphs=graphs_f[1:N-1],
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'reac_dens_f.png'),
    subtitles=['$\hat{\mu}^{AB}(' + str(n) + ')$' for n in np.arange(1, N-1)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_f[:N-1],
    shifted_rate=shifted_rate_f,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(charts_path, example_name + '_' + 'eff_f.png'),
    subtitles=['$f^+(' + str(n) + ')$' for n in np.arange(N-1)]
)
plot_rate(
    rate=rate_f,
    time_av_rate=time_av_rate_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_f.png'),
    title='Discrete finite-time rates',xlabel = 'n', average_rate_legend=r'$\bar{k}^{AB}_N$'
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_f.png'),
    title='Discrete finite-time reactiveness',
)


# plotting results for finite-time, time-inhomogeneous case
graphs_inhom = [nx.Graph(P_inhom(n)) for n in np.arange(N_inhom)] 
shifted_rate_inhom = np.zeros((N_inhom-1, 2))
shifted_rate_inhom[:, 0] = rate_inhom[:N_inhom-1, 0]
shifted_rate_inhom[:, 1] = rate_inhom[1:, 1]

plot_density(
    data=stat_dens_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'dens_inhom.png'),
    subtitles=['$\lambda(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=q_f_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_f_inhom.png'),
    subtitles=['$q^+(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=q_b_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(charts_path, example_name + '_' + 'q_b_inhom.png'),
    subtitles=['$q^-(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=norm_reac_dens_inhom[1:N_inhom-1],
    graphs=graphs_inhom[1:N_inhom-1],
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(charts_path, example_name + '_' + 'reac_dens_inhom.png'),
    subtitles=['$\hat{\mu}^{AB}(' + str(n) + ')$' for n in np.arange(1, N_inhom-1)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_inhom[:N_inhom-1],
    shifted_rate=shifted_rate_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(charts_path, example_name + '_' + 'eff_inhom.png'),
    subtitles=['$f^+(' + str(n) + ')$' for n in np.arange(N_inhom-1)]
)
plot_rate(
    rate=rate_inhom,
    time_av_rate=time_av_rate_inhom,
    file_path=os.path.join(charts_path, example_name + '_' + 'rates_inhom.png'),
    title='Discrete finite-time, time-inhomogeneous rates',xlabel = 'N', average_rate_legend=r'$\bar{k}^{AB}_n$'
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_inhom,
    file_path=os.path.join(charts_path, example_name + '_' + 'reactiveness_inhom.png'),
    title='Discrete finite-time, time-inhomogeneous reactiveness',
)

# extended finite-time -> large N=100
plot_convergence(
    q_f=q_f,
    q_f_conv=q_f_conv,
    q_b=q_b,
    q_b_conv=q_b_conv,
    scale_type='log',
    file_path=os.path.join(charts_path, example_name + '_' + 'conv_finite.png'),
    title=None,#'Convergence of finite-time, stationary $q^+(n)$ and $q^-(n)$ on $\{-N,...,N\}$ for large $N$',
)

