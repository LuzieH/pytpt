import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os.path
import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf
################# TODO
# plotting of directed weighted graphs 
# take out current density -> instead plot effective current (directed network)
# add colorbar to plots

###############################################################################
## load data about small network
#my_path = os.path.abspath(os.path.dirname(__file__))
#A = np.load(os.path.join(my_path,'data/small_network_A.npy'))
#T = np.load(os.path.join(my_path,'data/small_network_T.npy'))
#T_p = np.load(os.path.join(my_path,'data/small_network_T_periodisch.npy'))
#pos = np.load(os.path.join(my_path,'data/small_network_pos.npy'))
#pos = pos.item()
#labels = np.load(os.path.join(my_path,'data/small_network_labels.npy'))
#labels = labels.item()

######################################################## plotting
 
def plot_subplot(data, graph, pos, timeframe, size, v_min ,v_max, title , subtitles=None):
    fig, ax = plt.subplots(1,timeframe, sharex='col', sharey='row', figsize =size)
    if timeframe==1: 
        nx.draw(graph, pos=pos, labels=labels, node_color=data,ax=ax, vmin = v_min, vmax = v_max)  
    else:
        for i in range(timeframe):
            nx.draw(graph, pos=pos, labels=labels, node_color=data[i] ,ax=ax[i], vmin = v_min, vmax = v_max)
            if subtitles is not None: 
                ax[i].set_title(subtitles[i], pad=0)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)  
    return fig

######################################################## plotting
 
def plot_subplot_directed(weights, graph, pos, timeframe, size, v_min ,v_max, title, subtitles=None):
    fig, ax = plt.subplots(1,timeframe, sharex='col', sharey='row', figsize =size)
    if timeframe==1: 
            A_eff = (weights>0)*1
            G_eff = nx.DiGraph(A_eff)
            nbr_edges= int(np.sum(A_eff))
            edge_colors = np.zeros(nbr_edges)
            widths = np.zeros(nbr_edges)
            
            for j in np.arange(nbr_edges):
                edge_colors[j] = 200*weights[np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
                widths[j] = edge_colors[j] #weights[i,np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
            
            nx.draw_networkx_nodes(G_eff, pos, ax=ax)
            nx.draw_networkx_edges(G_eff, pos ,ax=ax, arrowsize=10, edge_color=edge_colors,width = widths,
                                           edge_cmap=plt.cm.Blues)
            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels ,ax=ax)
            ax.set_axis_off()
    else:
        for i in range(timeframe):
            A_eff = (weights[i,:,:]>0)*1
            G_eff = nx.DiGraph(A_eff)
            nbr_edges= int(np.sum(A_eff))
            edge_colors = np.zeros(nbr_edges)
            widths = np.zeros(nbr_edges)
            
            for j in np.arange(nbr_edges):
                edge_colors[j] = weights[i,np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
                widths[j] = 150*edge_colors[j] #weights[i,np.array(G_eff.edges())[j,0], np.array(G_eff.edges())[j,1]]
            
            nx.draw_networkx_nodes(G_eff, pos, ax=ax[i])
            nx.draw_networkx_edges(G_eff, pos ,ax=ax[i] , arrowsize=10, edge_color=edge_colors,width = widths,
                                           edge_cmap=plt.cm.Blues)
            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels ,ax=ax[i] )
            #ax = plt.gca()
            ax[i].set_axis_off()
            if subtitles is not None: 
                ax[i].set_title(subtitles[i], pad=0)
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
        
    return fig

###################################################################
## draw weighted, directed network with effective currente on edges
## first construct a subgraph of only the edges with non zero effective current
#plt.figure()
#plt.title('Effective current')
#A_eff = (eff_current>0)*1
#G_eff = nx.DiGraph(A_eff)
#nbr_edges= int(np.sum(A_eff))
#edge_colors = np.zeros(nbr_edges)
#widths = np.zeros(nbr_edges)
#for i in np.arange(nbr_edges):
#    edge_colors[i] = eff_current[np.array(G_eff.edges())[i,0], np.array(G_eff.edges())[i,1]]
#    widths[i] = 500*eff_current[np.array(G_eff.edges())[i,0], np.array(G_eff.edges())[i,1]]
#nodes = nx.draw_networkx_nodes(G_eff, pos)
#edges = nx.draw_networkx_edges(G_eff, pos,   arrowsize=10, edge_color=edge_colors,width = widths,
#                               edge_cmap=plt.cm.Blues)
## labels
#nx.draw_networkx_labels(G_eff, pos, font_size=20)
#ax = plt.gca()
#ax.set_axis_off()
#plt.show()
#

###########################################################general
G = nx.Graph(A)  
ind_A= np.array([0])
ind_B = np.array([1])
ind_C = np.arange(2,np.shape(T)[0])

##################### TPT ergodic, infinite time

#instantiate
small =tp.transitions_mcs(T+T_p, ind_A, ind_B, ind_C)
stat_dens = small.stationary_density()

#compute committor probabilities
[q_f,q_b] = small.committor()

#therof compute the reactive density
reac_dens = small.reac_density()

#and reactive currents
[current,eff_current] = small.reac_current()
current_dens = small.current_density()




################################################ TPT finite time
# transition matrix at time n
def Tn(n):
    return T+T_p

# initial density
init_dens_small =  stat_dens # 1./np.shape(T)[0]*np.ones(np.shape(T)[0])
N = 6 # size of time interval

# instantiate
small_finite = tpf.transitions_finite_time(Tn, N, ind_A, ind_B,  ind_C, init_dens_small)
[q_f_finite,q_b_finite] = small_finite.committor()

stat_dens_finite = small_finite.density()
# reactive density (zero at time 0 and N)
reac_dens_finite = small_finite.reac_density()

# and reactive currents
[current_finite,eff_current_finite] = small_finite.reac_current()
current_dens_finite = small_finite.current_density()

rate_finite = small_finite.transition_rate()


############################################## TPT periodisch
# use as transition matrix T+w T_p, where w varies from 0..1..0..-1..
#either faster switching or slower dynamics
M = 10 #6 # size of period

# transition matrix at time k
def Tm(k):
    # varies the transition matrices periodically, by weighting the added
    # matrix T_p with weights 0..1..0..-1.. over one period
    T_m = T + np.sin(k*2.*np.pi/M)*T_p
    return T_m

# instantiate   
small_periodic = tpp.transitions_periodic(Tm, M, ind_A, ind_B, ind_C)
stat_dens_p = small_periodic.stationary_density()

[q_f_p,q_b_p] = small_periodic.committor()
P_back_m = small_periodic.backward_transitions()
# reactive density 
reac_dens_p =small_periodic.reac_density()

# and reactive currents
[current_p,eff_current_p] = small_periodic.reac_current()
current_dens_p = small_periodic.current_density()

rate_p = small_periodic.transition_rate()

###################################plotting

v_min_dens = min([np.min(stat_dens),np.min(stat_dens_finite), np.min(stat_dens_p)])
v_max_dens = max([np.max(stat_dens),np.max(stat_dens_finite), np.max(stat_dens_p)])

v_min_reac_dens = min([np.min(reac_dens),np.min(reac_dens_finite), np.min(reac_dens_p)])
v_max_reac_dens= max([np.max(reac_dens),np.max(reac_dens_finite), np.max(reac_dens_p)])


v_min_current_dens  = min([np.min(current_dens),np.min(current_dens_finite), np.min(current_dens_p)])
v_max_current_dens  = max([np.max(current_dens),np.max(current_dens_finite), np.max(current_dens_p)])

 


## collect computed statistics for plotting
#C = 5
#data_coll = np.zeros((5,np.shape(stat_dens)[0])) 
#data_coll[0,:] = stat_dens
#data_coll[1,:] = q_f
#data_coll[2,:] = q_b
#data_coll[3,:] = reac_dens
#data_coll[4,:] = current_dens
#subtitles_coll = np.array(['Stationary density','$q^+$','$q^-$','Reactive density','Current density'])
#
#fig = plot_subplot(data_coll, G, pos, C, (2*C, 3),'Stationary system',subtitles_coll)

 
fig = plot_subplot( stat_dens , G, pos, 1, (2*1, 2),v_min_dens,v_max_dens,'Stationary density')
fig.savefig('dens.png', dpi=100)
fig = plot_subplot(q_f, G, pos, 1, (2*1, 2),0,1,'Forward committor')
fig.savefig('q_f.png', dpi=100)
fig = plot_subplot(q_b, G, pos, 1, (2*1, 2), 0,1,'Backward committor')
fig.savefig('q_b.png', dpi=100)
fig = plot_subplot(reac_dens, G, pos, 1, (2*1, 2),v_min_reac_dens,v_max_reac_dens, 'Reactive distribution')
fig.savefig('reac_dens.png', dpi=100)
fig = plot_subplot_directed(eff_current, G, pos, 1, (2*1, 2), 0,1,'Effective current')
fig.savefig('eff.png', dpi=100)
#fig = plot_subplot(current_dens, G, pos, 1, (2*1, 3),v_min_current_dens,v_max_current_dens, 'Current density')



# plotting
subtitles_p = np.array(['m = ' + str(i) for i in np.arange(M)])
fig = plot_subplot(stat_dens_p, G, pos, M, (2*M, 2),v_min_dens,v_max_dens,'Periodic stationary density',subtitles_p)
fig.savefig('dens_p.png', dpi=100)
fig = plot_subplot(q_f_p, G, pos, M, (2*M, 2),0,1,'Periodic forward committor',subtitles_p)
fig = plot_subplot(q_b_p, G, pos, M, (2*M, 2), 0,1,'Periodic backward committor',subtitles_p)
fig = plot_subplot(reac_dens_p, G, pos, M, (2*M, 2),v_min_reac_dens,v_max_reac_dens, 'Periodic reactive distribution',subtitles_p)
fig.savefig('reac_dens_p.png', dpi=100)
#fig = plot_subplot(current_dens_p, G, pos, M, (2*M, 3),v_min_current_dens,v_max_current_dens, 'Periodic current density',subtitles_p)
fig = plot_subplot_directed(eff_current_p, G, pos, M, (2*M, 2),v_min_dens,v_max_dens,'Periodic effective current', subtitles_p)
fig.savefig('eff_p.png', dpi=100)

# plotting
subtitles_f = np.array(['n = ' + str(i) for i in np.arange(N)])
fig = plot_subplot(stat_dens_finite, G, pos, N, (2*N, 2),v_min_dens,v_max_dens,'Finite-time density',subtitles_f)
fig.savefig('dens_f.png', dpi=100)
fig = plot_subplot(q_f_finite, G, pos, N, (2*N, 2),0,1,'Finite-time forward committor',subtitles_f)
fig = plot_subplot(q_b_finite, G, pos, N, (2*N, 2), 0,1,'Finite-time backward committor',subtitles_f)
fig = plot_subplot(reac_dens_finite, G, pos, N, (2*N, 2), v_min_reac_dens,v_max_reac_dens,'Finite-time reactive distribution',subtitles_f)
fig.savefig('reac_dens_f.png', dpi=100)
#fig = plot_subplot(current_dens_finite, G, pos, N, (2*N, 3), v_min_current_dens,v_max_current_dens,'Finite-time current density',subtitles_f)
fig = plot_subplot_directed(eff_current_finite, G, pos, N, (2*N, 2),v_min_dens,v_max_dens,'Finite-time effective current', subtitles_f)
fig.savefig('eff_f.png', dpi=100)

#########################extended finite-time -> large N=100
#fig = plot_subplot( stat_dens_e , G, pos, 1, (2*1, 2),v_min_dens,v_max_dens,'Density, n=0')
#fig.savefig('dens_ext.png', dpi=100)
#fig = plot_subplot(q_f_e, G, pos, 1, (2*1, 2),0,1,'Forward committor, n=0')
#fig.savefig('q_f_ext.png', dpi=100)
#fig = plot_subplot(q_b_e, G, pos, 1, (2*1, 2), 0,1,'Backward committor, n=0')
#fig.savefig('q_b_ext.png', dpi=100)
#fig = plot_subplot(reac_dens_e, G, pos, 1, (2*1, 2),v_min_reac_dens,v_max_reac_dens, 'Reactive distribution, n=0')
#fig.savefig('reac_dens_ext.png', dpi=100)
#fig = plot_subplot_directed(eff_current, G, pos, 1, (2*1, 2), 0,1,'Effective current, n=0')
#fig.savefig('eff_ext.png', dpi=100)
##fig = plot_subplot(current_dens, G, pos, 1, (2*1, 3),v_min_current_dens,v_max_current_dens, 'Current density')
