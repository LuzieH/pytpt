import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import os.path

# TODO add colorbar to plots

# general

# load data about small network
my_path = os.path.abspath(os.path.dirname(__file__))
pos = np.load(
    os.path.join(my_path, 'data/small_network_pos.npy'),
    allow_pickle=True,
)
pos = pos.item()
labels = np.load(
    os.path.join(my_path, 'data/small_network_labels.npy'),
    allow_pickle=True, 
)
labels = labels.item()
T = np.load(os.path.join(my_path, 'data/small_network_T.npy'))
L = np.load(os.path.join(my_path, 'data/small_network_L.npy'))
K = np.load(os.path.join(my_path, 'data/small_network_K.npy'))

ind_A = np.array([0])
ind_C = np.arange(1, np.shape(T)[0] - 1)
ind_B = np.array([4])


# TPT ergodic, infinite-time

# transition matrix at time n
def P(n):
    return T + L

# instantiate
small = tp.transitions_mcs(T + L, ind_A, ind_B, ind_C)
stat_dens = small.stationary_density()

# compute committor probabilities
[q_f, q_b] = small.committor()

# therof compute the normalized reactive density
norm_reac_dens = small.norm_reac_density()

# and reactive currents
[current, eff_current] = small.reac_current()
rate = small.transition_rate()  # AB discrete transition rate


# TPT finite time, time-homogeneous

# transition matrix at time n
def P_hom(n):
    return T + L

# initial density
init_dens_small = stat_dens
N = 6  # size of time interval

# instantiate
small_finite = tpf.transitions_finite_time(
    P_hom, N, ind_A, ind_B,  ind_C, init_dens_small)
[q_f_finite, q_b_finite] = small_finite.committor()

stat_dens_finite = small_finite.density()
# reactive density (zero at time 0 and N)
norm_reac_dens_finite = small_finite.norm_reac_density()

# and reactive currents
[current_finite, eff_current_finite] = small_finite.reac_current()

# first row, out rate of A, second row in rate for B
rate_finite = small_finite.transition_rate()


# TPT finite time, time-inhomogeneous

# transition matrix at time n
def P_inhom(n):
    if n == 0: 
        return T + L + 3*K
    elif n == 1: 
        return T + L + 2*K
    elif n == 2: 
        return T + L + K
    else:
        return T + L

# initial density
init_dens_small_inhom = stat_dens
N = 6  # size of time interval

# instantiate
small_finite_inhom = tpf.transitions_finite_time(
    P_inhom, N, ind_A, ind_B,  ind_C, init_dens_small_inhom)
[q_f_finite_inhom, q_b_finite_inhom] = small_finite_inhom.committor()

stat_dens_finite_inhom = small_finite_inhom.density()
# reactive density (zero at time 0 and N)
norm_reac_dens_finite_inhom = small_finite_inhom.norm_reac_density()

# and reactive currents
[current_finite_inhom, eff_current_finite_inhom] = small_finite_inhom.reac_current()

# first row, out rate of A, second row in rate for B
rate_finite_inhom = small_finite_inhom.transition_rate()



# TPT finite time extension to infinite time, convergence analysis

N_ex = 150  # size of time interval
q_f_conv = np.zeros((N_ex-1, np.shape(T)[0]))
for ne in np.arange(1, N_ex):
    # instantiate
    small_finite_ex = tpf.transitions_finite_time(
        P_hom, ne*2+1, ind_A, ind_B,  ind_C, init_dens_small)
    [q_f_ex, q_b_ex] = small_finite_ex.committor()
    q_f_conv[ne-1, :] = q_f_ex[ne, :]


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

rate_p = small_periodic.transition_rate()


# plotting

def plot_density(data, P, pos, timeframe, size,
                 v_min, v_max, title, subtitles=None):
    fig, ax = plt.subplots(1, timeframe, sharex='col',
                           sharey='row', figsize=size)
    if timeframe == 1:
        graph = nx.Graph(P(timeframe))
        nx.draw(graph, pos=pos, labels=labels,
                node_color=data, ax=ax, vmin=v_min, vmax=v_max)
    else:
        for i in range(timeframe):
            graph = nx.Graph(P(i))
            nx.draw(graph, pos=pos, labels=labels,
                    node_color=data[i], ax=ax[i], vmin=v_min, vmax=v_max)
            if subtitles is not None:
                ax[i].set_title(subtitles[i])
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    return fig


def plot_effective_current(weights, graph, pos, timeframe, size, v_min, v_max, title, subtitles=None):
    fig, ax = plt.subplots(1, timeframe, sharex='col',
                           sharey='row', figsize=size)
    if timeframe == 1:
        A_eff = (weights > 0)*1
        G_eff = nx.DiGraph(A_eff)
        nbr_edges = int(np.sum(A_eff))
        edge_colors = np.zeros(nbr_edges)
        widths = np.zeros(nbr_edges)

        for j in np.arange(nbr_edges):
            edge_colors[j] = 200 * \
                weights[np.array(G_eff.edges())[j, 0],
                        np.array(G_eff.edges())[j, 1]]
            # weights[i,np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
            widths[j] = edge_colors[j]

        nx.draw_networkx_nodes(G_eff, pos, ax=ax)
        nx.draw_networkx_edges(G_eff, pos, ax=ax, arrowsize=10, edge_color=edge_colors, width=widths,
                               edge_cmap=plt.cm.Blues)
        # labels
        nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax)
        ax.set_axis_off()
    else:
        for i in range(timeframe):
            A_eff = (weights[i, :, :] > 0)*1
            G_eff = nx.DiGraph(A_eff)
            nbr_edges = int(np.sum(A_eff))
            edge_colors = np.zeros(nbr_edges)
            widths = np.zeros(nbr_edges)

            for j in np.arange(nbr_edges):
                edge_colors[j] = weights[i, np.array(
                    G_eff.edges())[j, 0], np.array(G_eff.edges())[j, 1]]
                # weights[i,np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
                widths[j] = 150*edge_colors[j]

            nx.draw_networkx_nodes(G_eff, pos, ax=ax[i])
            nx.draw_networkx_edges(G_eff, pos, ax=ax[i], arrowsize=10, edge_color=edge_colors, width=widths,
                                   edge_cmap=plt.cm.Blues)
            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax[i])
            #ax = plt.gca()
            ax[i].set_axis_off()
            if subtitles is not None:
                ax[i].set_title(subtitles[i])  # , pad=0)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)

    return fig

#########################################################


v_min_dens = min([
    np.min(stat_dens),
    np.min(stat_dens_finite),
    np.min(stat_dens_p),
])
v_max_dens = max([
    np.max(stat_dens),
    np.max(stat_dens_finite),
    np.max(stat_dens_p),
])
v_min_reac_dens = min([
    np.min(norm_reac_dens),
    np.min(norm_reac_dens_finite),
    np.min(norm_reac_dens_p),
])
v_max_reac_dens = max([
    np.max(norm_reac_dens),
    np.max(norm_reac_dens_finite),
    np.max(norm_reac_dens_p)],
)

# collect computed statistics for plotting
#C = 5
#data_coll = np.zeros((5,np.shape(stat_dens)[0]))
#data_coll[0,:] = stat_dens
#data_coll[1,:] = q_f
#data_coll[2,:] = q_b
#data_coll[3,:] = reac_dens
#subtitles_coll = np.array(['Stationary density','$q^+$','$q^-$','Reactive density','Current density'])
#
#fig = plot_subplot(data_coll, G, pos, C, (2*C, 3),'Stationary system',subtitles_coll)

G = nx.Graph(T+L)

# define directory path to save the plots
charts_path = os.path.join(my_path, 'charts')

# plotting results for ergodic, infinite-time case
fig = plot_density(stat_dens, P, pos, 1, (2*1, 2),
                   v_min_dens, v_max_dens, 'Stationary density')

fig.savefig(os.path.join(charts_path, 'dens.png'), dpi=100)

fig = plot_density(q_f, P, pos, 1, (2*1, 2), 0, 1, '$q^+$')
fig.savefig(os.path.join(charts_path, 'q_f.png'), dpi=100)

fig = plot_density(q_b, P, pos, 1, (2*1, 2), 0, 1, '$q^-$')
fig.savefig(os.path.join(charts_path, 'q_b.png'), dpi=100)

fig = plot_density(norm_reac_dens, P, pos, 1, (2*1, 2),
                   v_min_reac_dens, v_max_reac_dens, '$\mu^\mathcal{AB}$')
fig.savefig(os.path.join(charts_path, 'reac_dens.png'), dpi=100)


fig = plot_effective_current(eff_current, G, pos, 1,
                            (2*1, 2), 0, 1, 'Effective current $f^+$')
fig.savefig(os.path.join(charts_path, 'eff.png'), dpi=100)
#plt.close()


# plotting results for periodic case
subtitles_p = np.array(['m = ' + str(i) for i in np.arange(M)])
fig = plot_density(stat_dens_p, P_p, pos, M, (2*M, 2), v_min_dens,
                   v_max_dens, 'Periodic stationary density', subtitles_p)
fig.savefig(os.path.join(charts_path, 'dens_p.png'), dpi=100)

fig = plot_density(q_f_p, P_p, pos, M, (2*M, 2), 0, 1,
                   'Periodic $q^+_m$', subtitles_p)
fig.savefig(os.path.join(charts_path, 'q_f_p.png'), dpi=100)

fig = plot_density(q_b_p, P_p, pos, M, (2*M, 2), 0, 1,
                   'Periodic $q^-_m$', subtitles_p)
fig.savefig(os.path.join(charts_path, 'q_b_p.png'), dpi=100)

fig = plot_density(norm_reac_dens_p, P_p, pos, M, (2*M, 2), v_min_reac_dens,
                   v_max_reac_dens, 'Periodic $\mu_m^\mathcal{AB}$', subtitles_p)
fig.savefig(os.path.join(charts_path, 'reac_dens_p.png'), dpi=100)

fig = plot_effective_current(eff_current_p, G, pos, M, (2*M, 2), v_min_dens,
                            v_max_dens, 'Periodic effective current $f^+_m$', subtitles_p)
fig.savefig(os.path.join(charts_path, 'eff_p.png'), dpi=100)
#plt.close()


# plotting results for finite-time, time-homogeneous case
subtitles_f = np.array(['n = ' + str(i) for i in np.arange(N)])
fig = plot_density(stat_dens_finite, P_hom, pos, N, (2*N, 2),
                   v_min_dens, v_max_dens, 'Finite-time density', subtitles_f)
fig.savefig(os.path.join(charts_path, 'dens_f.png'), dpi=100)

fig = plot_density(q_f_finite, P_hom, pos, N, (2*N, 2), 0, 1,
                   'Finite-time $q^+(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'q_f_f.png'), dpi=100)

fig = plot_density(q_b_finite, P_hom, pos, N, (2*N, 2), 0, 1,
                   'Finite-time $q^-(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'q_b_f.png'), dpi=100)

fig = plot_density(norm_reac_dens_finite, P_hom, pos, N, (2*N, 2), v_min_reac_dens,
                   v_max_reac_dens, 'Finite-time $\mu^\mathcal{AB}(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'reac_dens_f.png'), dpi=100)



fig = plot_effective_current(eff_current_finite, G, pos, N, (2*N, 2), v_min_dens, v_max_dens, 'Finite-time $f^+(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'eff_f.png'), dpi=100)
#plt.close()


# plotting results for finite-time, time-inhomogeneous case
subtitles_f = np.array(['n = ' + str(i) for i in np.arange(N)])
fig = plot_density(stat_dens_finite, P_inhom, pos, N, (2*N, 2),
                   v_min_dens, v_max_dens, 'Finite-time density', subtitles_f)
fig.savefig(os.path.join(charts_path, 'dens_f_inhom.png'), dpi=100)

fig = plot_density(q_f_finite, P_inhom, pos, N, (2*N, 2), 0, 1,
                   'Finite-time $q^+(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'q_f_inhom.png'), dpi=100)

fig = plot_density(q_b_finite, P_inhom, pos, N, (2*N, 2), 0, 1,
                   'Finite-time $q^-(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'q_b_inhom.png'), dpi=100)

fig = plot_density(norm_reac_dens_finite, P_inhom, pos, N, (2*N, 2), v_min_reac_dens,
                   v_max_reac_dens, 'Finite-time $\mu^\mathcal{AB}(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'reac_dens_f_inhom.png'), dpi=100)

fig = plot_effective_current(eff_current_finite, G, pos, N, (2*N, 2), v_min_dens, v_max_dens, 'Finite-time $f^+(n)$', subtitles_f)
fig.savefig(os.path.join(charts_path, 'eff_f_inhom.png'), dpi=100)
#plt.close()


# extended finite-time -> large N=100
fig, ax = plt.subplots(1, 1, figsize=(2*M, 5))
convergence_error = np.linalg.norm(q_f_conv - q_f, ord=2, axis=1)
plt.plot(np.arange(1, N_ex), convergence_error)  # , s=5, marker='o')
plt.title(
    'Convergence of finite-time, stationary $q^+(n)$ on $\{-N,...,N\}$ for large $N$')
plt.xlabel('$N$')
plt.ylabel('$l_2$-Error $||q^+ - q^+(0)||$ ')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.savefig(os.path.join(charts_path, 'conv_finite.png'), dpi=100)

# rates
# periodic

fig, ax = plt.subplots(1, 1, figsize=(2*M, 2))
#fig = plt.figure(figsize=(2*M, 2))
# $k^{A \rightarrow}(m)$
plt.scatter(np.arange(M), rate_p[0, :], label='$k^{A->}$', alpha=0.7)
plt.scatter(np.arange(M), rate_p[1, :], label='$k^{->B}$', alpha=0.7)
plt.legend()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title('Discrete M-periodic rates')
plt.xlabel('m')
plt.ylabel('Discrete rate')
fig.savefig(os.path.join(charts_path, 'rates_p.png'), dpi=100)

fig, ax = plt.subplots(1, 1, figsize=(2*N, 2))
plt.scatter(np.arange(N), rate_finite[0, :], label='$k^{A->}$', alpha=0.7)
plt.scatter(np.arange(N), rate_finite[1, :], label='$k^{->B}$', alpha=0.7)
plt.legend()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title('Discrete finite-time rates')
plt.xlabel('n')
plt.ylabel('Discrete rate')
fig.savefig(os.path.join(charts_path, 'rates_finite.png'), dpi=100)
