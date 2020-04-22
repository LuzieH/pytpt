from small_network_example import P_p, P_hom, P_inhom

from plotting import plot_network_density as plot_density, \
                     plot_network_effective_current as plot_effective_current, \
                     plot_network_effcurrent_and_rate as plot_effcurrent_and_rate, \
                     plot_rate, \
                     plot_reactiveness, \
                     plot_convergence, \
                     plot_colorbar_only


import functools
import numpy as np
import networkx as nx

import os

# define directories path to save the data and figures 
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, 'data')
figures_path = os.path.join(my_path, 'figures')

# load small network construction data
network_construction = np.load(
    os.path.join(data_path, 'small_network_construction.npz'),
    allow_pickle=True,
)
states = network_construction['states'].item()
labels = network_construction['labels'].item()
pos = network_construction['pos'].item()
T = network_construction['T']
L = network_construction['L']
K = network_construction['K']
P = T + L

# load small network statistics data 
example_name = 'small_network_stationary'
network_ergodic = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'small_network_periodic'
network_periodic = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'small_network_finite'
network_finite = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'small_network_inhom'
network_inhom = np.load(
    os.path.join(data_path, example_name + '.npz')
)
example_name = 'small_network_conv'
network_conv = np.load(
    os.path.join(data_path, example_name + '.npz')
)

stat_dens = network_ergodic['stat_dens']
q_f = network_ergodic['q_f']
q_b = network_ergodic['q_b']
reac_norm_factor = network_ergodic['reac_norm_factor']
norm_reac_dens = network_ergodic['norm_reac_dens']
eff_current = network_ergodic['eff_current']
rate = network_ergodic['rate']
length = network_ergodic['length']

stat_dens_p = network_periodic['stat_dens']
q_f_p = network_periodic['q_f']
q_b_p = network_periodic['q_b']
reac_norm_factor_p = network_periodic['reac_norm_factor']
norm_reac_dens_p = network_periodic['norm_reac_dens']
eff_current_p = network_periodic['eff_current']
rate_p = network_periodic['rate']
time_av_rate_p = network_periodic['time_av_rate']
av_length_p = network_periodic['av_length']

dens_f = network_finite['dens']
q_f_f = network_finite['q_f']
q_b_f = network_finite['q_b']
reac_norm_factor_f = network_finite['reac_norm_factor']
norm_reac_dens_f = network_finite['norm_reac_dens']
eff_current_f = network_finite['eff_current']
rate_f = network_finite['rate']
time_av_rate_f = network_finite['time_av_rate']
av_length_f = network_finite['av_length']

dens_inhom = network_inhom['dens']
q_f_inhom = network_inhom['q_f']
q_b_inhom = network_inhom['q_b']
reac_norm_factor_inhom = network_inhom['reac_norm_factor']
norm_reac_dens_inhom = network_inhom['norm_reac_dens']
eff_current_inhom = network_inhom['eff_current']
rate_inhom = network_inhom['rate']
time_av_rate_inhom = network_inhom['time_av_rate']
av_length_inhom = network_inhom['av_length']

q_f_conv = network_conv['q_f']
q_b_conv = network_conv['q_b']


print("rate (infinite-time, stationary): %f" % rate)
print("periodic-averaged rate (infinite-time, periodic): %f" % time_av_rate_p[0])
print("time-averaged rate (finite-time, time-hom.): %f" % time_av_rate_f[0])
print("time-averaged rate (finite-time, time-inhom.): %f" % time_av_rate_inhom[0])

print("mean length (infinite-time, stationary): %f" % length)
print("mean length (infinite-time, periodic): %f" % av_length_p)
print("mean length (finite-time, time-homogeneous): %f" % av_length_f)
print("mean length (finite-time, time-inhomogeneous): %f" % av_length_inhom)

v_min_dens = min([
    np.min(stat_dens),
    np.min(stat_dens_p),
    np.min(dens_f),
    np.min(dens_inhom),
])
v_max_dens = max([
    np.max(stat_dens),
    np.max(stat_dens_p),
    np.max(dens_f),
    np.max(dens_inhom),
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


# plot relative color bar
plot_colorbar_only(
    file_path=os.path.join(figures_path, 'small_network_colorbar.png'),
)

# plotting results for ergodic, infinite-time case
example_name = 'small_network_stationary'
graphs = [nx.Graph(P)]

plot_density(
    data=np.array([stat_dens]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'dens.png'),
    subtitles=['$\pi$']
)
plot_density(
    data=np.array([q_f]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_f.png'),
    subtitles=['$q^+$'],
)
plot_density(
    data=np.array([q_b]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_b.png'),
    subtitles=['$q^-$'],
)
plot_density(
    data=np.array([norm_reac_dens]),
    graphs=graphs,
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'reac_dens.png'),
    subtitles=['$\hat{\mu}^{AB}$'],
)
plot_effcurrent_and_rate(
    eff_current=np.array([eff_current]),
    shifted_rate=[[rate, rate]],
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(figures_path, example_name + '_' + 'eff.png'),
    subtitles=['$f^+$'],
)

# plotting results for periodic case
example_name = 'small_network_periodic'
M = 6
P_p = functools.partial(P_p, M=M)
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
    file_path=os.path.join(figures_path, example_name + '_' + 'dens.png'),
    subtitles=['$\pi_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=q_f_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_f.png'),
    subtitles=['$q^+_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=q_b_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_b.png'),
    subtitles=['$q^-_' + str(m) + '$' for m in np.arange(M)]
)
plot_density(
    data=norm_reac_dens_p,
    graphs=graphs_p,
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'reac_dens.png'),
    subtitles=['$\hat{\mu}_' + str(m) + '^{AB}$' for m in np.arange(M)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_p,
    shifted_rate=shifted_rate_p,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(figures_path, example_name + '_' + 'eff.png'),
    subtitles=['$f^+_' + str(m) + '$' for m in np.arange(M)]
)
plot_rate(
    rate=rate_p,
    file_path=os.path.join(figures_path, example_name + '_' + 'rates.png'),
    title='Discrete M-periodic rates',
    xlabel = 'm',
    average_rate_legend=r'$\bar{k}^{AB}_M$',
)

# plotting results for finite-time, time-homogeneous case
example_name = 'small_network_finite'
N = 5
graphs_f = [nx.Graph(P_hom(n)) for n in np.arange(N)] 
shifted_rate_f = np.zeros((N-1, 2))
shifted_rate_f[:, 0] = rate_f[:N-1, 0]
shifted_rate_f[:, 1] = rate_f[1:, 1]

plot_density(
    data=dens_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'dens.png'),
    subtitles=['$\pi(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=q_f_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_f.png'),
    subtitles=['$q^+(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=q_b_f,
    graphs=graphs_f,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_b.png'),
    subtitles=['$q^-(' + str(n) + ')$' for n in np.arange(N)]
)
plot_density(
    data=norm_reac_dens_f[1:N-1],
    graphs=graphs_f[1:N-1],
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'reac_dens.png'),
    subtitles=['$\hat{\mu}^{AB}(' + str(n) + ')$' for n in np.arange(1, N-1)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_f[:N-1],
    shifted_rate=shifted_rate_f,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(figures_path, example_name + '_' + 'eff.png'),
    subtitles=['$f^+(' + str(n) + ')$' for n in np.arange(N-1)]
)
plot_rate(
    rate=rate_f,
    time_av_rate=time_av_rate_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'rates.png'),
    title='Discrete finite-time rates',
    xlabel = 'n',
    average_rate_legend=r'$\bar{k}^{AB}_N$',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'reactiveness.png'),
    title='Discrete finite-time reactiveness',
)


# plotting results for finite-time, time-inhomogeneous case
example_name = 'small_network_inhom'
N_inhom = 5
graphs_inhom = [nx.Graph(P_inhom(n)) for n in np.arange(N_inhom)] 
shifted_rate_inhom = np.zeros((N_inhom-1, 2))
shifted_rate_inhom[:, 0] = rate_inhom[:N_inhom-1, 0]
shifted_rate_inhom[:, 1] = rate_inhom[1:, 1]

plot_density(
    data=dens_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_dens,
    v_max=v_max_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'dens.png'),
    subtitles=['$\lambda(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=q_f_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_q_f,
    v_max=v_max_q_f,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_f.png'),
    subtitles=['$q^+(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=q_b_inhom,
    graphs=graphs_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_q_b,
    v_max=v_max_q_b,
    file_path=os.path.join(figures_path, example_name + '_' + 'q_b.png'),
    subtitles=['$q^-(' + str(n) + ')$' for n in np.arange(N_inhom)]
)
plot_density(
    data=norm_reac_dens_inhom[1:N_inhom-1],
    graphs=graphs_inhom[1:N_inhom-1],
    pos=pos,
    labels=labels,
    v_min=v_min_reac_dens,
    v_max=v_max_reac_dens,
    file_path=os.path.join(figures_path, example_name + '_' + 'reac_dens.png'),
    subtitles=['$\hat{\mu}^{AB}(' + str(n) + ')$' for n in np.arange(1, N_inhom-1)]
)
plot_effcurrent_and_rate(
    eff_current=eff_current_inhom[:N_inhom-1],
    shifted_rate=shifted_rate_inhom,
    pos=pos,
    labels=labels,
    v_min=v_min_eff_curr,
    v_max=v_max_eff_curr,
    file_path=os.path.join(figures_path, example_name + '_' + 'eff.png'),
    subtitles=['$f^+(' + str(n) + ')$' for n in np.arange(N_inhom-1)]
)
plot_rate(
    rate=rate_inhom,
    time_av_rate=time_av_rate_inhom,
    file_path=os.path.join(figures_path, example_name + '_' + 'rates.png'),
    title='Discrete finite-time, time-inhomogeneous rates',
    xlabel = 'N',
    average_rate_legend=r'$\bar{k}^{AB}_n$',
)
plot_reactiveness(
    reac_norm_factor=reac_norm_factor_inhom,
    file_path=os.path.join(figures_path, example_name + '_' + 'reactiveness.png'),
    title='Discrete finite-time, time-inhomogeneous reactiveness',
)

# extended finite-time -> large N=100
example_name = 'small_network_conv'
plot_convergence(
    q_f=q_f,
    q_f_conv=q_f_conv,
    q_b=q_b,
    q_b_conv=q_b_conv,
    scale_type='log',
    file_path=os.path.join(figures_path, example_name + '_' + 'conv.png'),
    title=None,#'Convergence of finite-time, stationary $q^+(n)$ and $q^-(n)$ on $\{-N,...,N\}$ for large $N$',
)
