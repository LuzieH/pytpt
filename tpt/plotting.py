import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import ScalarFormatter

PLASMA = mpl.cm.get_cmap('plasma', 512)
NETWORK_CMAP = mpl.colors.ListedColormap(
    PLASMA(np.linspace(0.40, 0.90, 256))
)

def plot_network_colorbar(v_min, v_max, file_path):
    fig, ax = plt.subplots(figsize=(0.15, 2))

    cmap = NETWORK_CMAP
    norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    cb1 = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='vertical',
    )
    cb1.ax.tick_params(length=4, width=0.5, labelsize=8)
    
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')  


def plot_network_density(data, graphs, pos, labels, v_min, v_max, file_path, title=None, subtitles=None):
    # TODO document method
    """
    plots bla bla
    
    parameters
    data : ndarray
        bla
    graphs : list
        bla
    pos :
        bla
    vmin :
        bla
    vmax :
        bla
    title :
        bla
    subtitles :
        bla
    file_path:
        bla
    """
    
    num_plots = len(graphs)
    width_plot = 2
    height_plot = 2
    size = (width_plot*num_plots, height_plot)
    
    fig, ax = plt.subplots(1, num_plots, sharex='col',
                           sharey='row', figsize=size)
    if num_plots == 1:
        ax = [ax]
    for i in range(num_plots):
        nx.draw(graphs[i], pos=pos, labels=labels, node_color=data[i],
                cmap=NETWORK_CMAP, ax=ax[i], vmin=v_min, vmax=v_max, alpha=0.8)
        if subtitles is not None:
            ax[i].set_title(subtitles[i])
    
    if title is not None:
        fig.suptitle(title)
    
    fig.subplots_adjust(top=0.8) 

    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight')  

    
def plot_network_effective_current(eff_current, pos, labels, v_min, v_max, file_path, title=None, subtitles=None):
    # TODO document method
    
    num_plots = len(eff_current)
    width_plot = 2
    height_plot = 2
    size = (width_plot*num_plots, height_plot)

    fig, ax = plt.subplots(1, num_plots, sharex='col',
                           sharey='row', figsize=size)
    if num_plots == 1:
        ax = [ax]

    for n in range(num_plots):
        if not np.isnan(eff_current[n]).any():

            # graph
            A_eff = (eff_current[n, :, :] > 0)*1
            G_eff = nx.DiGraph(A_eff)
            
            nx.draw_networkx_nodes(
                G_eff,
                pos,
                ax=ax[n],
                node_color='lightgrey',
            )

            # edges
            nbr_edges = int(np.sum(A_eff))
            edge_colors = np.zeros(nbr_edges)
            widths = np.zeros(nbr_edges)
            for j in np.arange(nbr_edges):
                edge_colors[j] = eff_current[
                    n,
                    np.array(G_eff.edges())[j, 0],
                    np.array(G_eff.edges())[j, 1],
                ]
                widths[j] = 150*edge_colors[j]
            nx.draw_networkx_edges(
                G_eff,
                pos,
                ax=ax[n],
                width=widths,
                edge_color=edge_colors,
                edge_cmap=plt.cm.Greys,
                arrowsize=10,
            )
            
            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax[n])
            ax[n].set_axis_off()

            if subtitles is not None:
                ax[n].set_title(subtitles[n])
    
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(top=0.8)

    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight') 


def plot_network_effcurrent_and_rate(eff_current, shifted_rate, pos, labels, v_min, v_max, file_path, title=None, subtitles=None):
    # TODO document method
    
    num_plots = len(eff_current)
    width_plot = 2
    height_plot = 2
    size = (width_plot*num_plots, height_plot)

    fig, ax = plt.subplots(1, num_plots, sharex='col',
                           sharey='row', figsize=size)
    if num_plots == 1:
        ax = [ax]

    for n in range(num_plots):
        if not np.isnan(eff_current[n]).any(): 

            A_eff = (eff_current[n, :, :] > 0)*1
            G_eff = nx.DiGraph(A_eff)
            G_eff.add_edge(0, 0)

            # nodes
            nx.draw_networkx_nodes(
                G_eff,
                pos,
                nodelist=[1, 2, 3],
                node_color='lightgrey',
                alpha=0.8,
                ax=ax[n],
            )
            nx.draw_networkx_nodes(
                G_eff,
                pos,
                nodelist=[0, 4],
                node_color=shifted_rate[n],
                cmap=NETWORK_CMAP,
                alpha=0.8,
                ax=ax[n],
            )

            # edges
            nbr_edges = len(G_eff.edges())
            edge_eff_current = np.zeros(nbr_edges)
            widths = np.zeros(nbr_edges)
            for j in np.arange(nbr_edges):
                edge_eff_current[j] = eff_current[
                    n,
                    np.array(G_eff.edges())[j, 0],
                    np.array(G_eff.edges())[j, 1],
                ]
            nx.draw_networkx_edges(
                G_eff,
                pos,
                ax=ax[n],
                width=edge_eff_current*150,
                edge_color=edge_eff_current*150,
                edge_cmap=plt.cm.Greys,
                edge_vmin=0,
                arrowsize=10,
                alpha=0.8,
            )
            #print(eff_current[0]) 
            #print(A_eff)
            #print(G_eff.edges())
            #print(edge_colors)
            #return
            
            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax[n])
            ax[n].set_axis_off()

            if subtitles is not None:
                ax[n].set_title(subtitles[n])
    
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(top=0.8)

    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight')  


def plot_rate(rate, file_path, title, xlabel, average_rate_legend='$\hat{k}^{AB}$', time_av_rate=None):                                          
    # TODO document method
    ncol = 2
    timeframes = len(rate)
    fig, ax = plt.subplots(1, 1, figsize=(4*timeframes, 3))
    
    plt.scatter(
        x=np.arange(timeframes),
        y=rate[:, 0],
        marker='.',
        s=100,
        color='black',
        label='$k^{A->}$',
    )
    plt.scatter(
        x=np.arange(timeframes),
        y=rate[:, 1],
        marker='*',
        s=100,
        edgecolors='black',
        facecolors='none',
        label='$k^{->B}$',
    )
    if type(time_av_rate) != type(None):
        ncol = 3
        ax.hlines(
            y=time_av_rate[0],
            xmin=np.arange(timeframes)[0],
            xmax=np.arange(timeframes)[-1],
            color='black',
            #s=20,
            linestyles='dashed',
            label=average_rate_legend,#'$\hat{k}^{AB}_M$',
        )

    # add title and legend
    plt.title(title, fontsize=20)
    min_rate = np.nanmin([
        np.nanmin(rate[:, 0]),
        np.nanmin(rate[:, 1]),
    ])
    max_rate = np.nanmax([
        np.nanmax(rate[:, 0]),
        np.nanmax(rate[:, 1]),
    ])
    plt.ylim(
        min_rate - (max_rate-min_rate)/4,
        max_rate + (max_rate-min_rate)/4,
    )
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Rate', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(ncol=ncol, fontsize=20)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight')  


def plot_reactiveness(reac_norm_factor, file_path, title):
    # TODO document method
    timeframes = len(reac_norm_factor)

    fig, ax = plt.subplots(1, 1, figsize=(5*timeframes, 5))

    plt.scatter(
        np.arange(timeframes),
        reac_norm_factor[:],
        color='black',
        s=20,
        label='$\sum_{j \in C} \mu_j^{R}(n)$',
    )

    plt.title(title, fontsize=20)
    min_norm_factor = np.nanmin(reac_norm_factor)
    max_norm_factor = np.nanmax(reac_norm_factor)
    plt.ylim(
        min_norm_factor - (max_norm_factor - min_norm_factor)/4,
        max_norm_factor + (max_norm_factor - min_norm_factor)/4,
    )
    plt.xlabel('$n$', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight') 


def plot_convergence(q_f, q_f_conv, q_b, q_b_conv, scale_type, file_path, title):
    # TODO document method
    assert scale_type in ['linear', 'log', 'symlog', 'logit']

    # compute errors 
    q_f_conv_error = np.linalg.norm(q_f_conv - q_f, ord=2, axis=1)
    q_b_conv_error = np.linalg.norm(q_b_conv - q_b, ord=2, axis=1)

    N_max = len(q_f_conv)

    fig, ax = plt.subplots(1, 1, figsize=(20,8))#(25, 5))

    plt.yscale(scale_type)
    plt.plot(
        np.arange(1, N_max + 1)[::2],
        q_f_conv_error[::2],
        marker='.',
        color='black',
        linestyle='None',
        label='$||q^+ - q^+(0)||_2$',
    )
    plt.plot(
        np.arange(1, N_max + 1)[::2],
        q_b_conv_error[::2],
        marker='o',
        markeredgecolor='black',
        markerfacecolor='None',
        linestyle='None',
        label='$||q^- - q^-(0)||_2$',
    )

    if title is not None:
        plt.title(title, fontsize=30)
    plt.xlabel('$N$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(ncol=2, fontsize=25)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight') 


def plot_3well_potential(potential, file_path, title, subtitles=None):
    # create mesh grid
    delta = 0.01
    x = np.arange(-2.0, 2.0 + delta, delta)
    y = np.arange(-1.0, 2.0 + delta, delta)
    X, Y = np.meshgrid(x, y)
    
    # compute potential on the grid
    potential = potential(X, Y) 
   
    number_of_plots = 1
    size = (4*number_of_plots, 3) 
    fig = plt.figure(figsize=size)

    grid = AxesGrid(
        fig,
        rect=111,
        nrows_ncols=(1, number_of_plots),
        axes_pad=0.13,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1,
    )    

    for i in range(number_of_plots):
        im = grid[i].imshow(
            potential,
            vmin=potential.min(),
            vmax=potential.max(),
            origin='lower',
            extent=[-2, 2, -1, 2],
        )
        grid[i].title.set_text(subtitles[i])
    
    # add color bar
    cbar_pot = grid[i].cax.colorbar(im)
    cbar_pot = grid.cbar_axes[0].colorbar(im)
    
    # save figure
    fig.subplots_adjust(top=0.8)
    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight') 

def plot_3well_vector_field(vector_field, vector_field_forced,
                                   file_path, title, subtitles=None):

    #create mesh grid 
    delta = 0.20
    x = np.arange(-2.0, 2.0 + delta, delta)
    y = np.arange(-1.0, 2.0 + delta, delta)
    X, Y = np.meshgrid(x, y)
    
    # compute gradient/forced gradient on the grid
    U, V = vector_field(X, Y) 
    U_forced_0, V_forced_0 = vector_field_forced(X, Y, 0) 
    U_forced_3, V_forced_3 = vector_field_forced(X, Y, 3) 

    norm = np.linalg.norm(np.array([U, V]), axis=0)
    norm_forced_0 = np.linalg.norm(np.array([U_forced_0, V_forced_0]), axis=0)
    norm_forced_3 = np.linalg.norm(np.array([U_forced_3, V_forced_3]), axis=0)
    
    U_norm = U/norm
    U_forced_0_norm = U_forced_0/norm_forced_0
    U_forced_3_norm = U_forced_3/norm_forced_3
    V_norm = V/norm
    V_forced_0_norm = V_forced_0/norm_forced_0
    V_forced_3_norm = V_forced_3/norm_forced_3

    Us = [U_norm, U_forced_0_norm, U_forced_3_norm]
    Vs = [V_norm, V_forced_0_norm, V_forced_3_norm]
    norms = [norm, norm_forced_0, norm_forced_3]

    number_of_plots = 3 
    size = (4*number_of_plots, 3) 
    fig = plt.figure(figsize=size)
    
    # create grid
    grid = AxesGrid(
        fig,
        rect=111,
        nrows_ncols=(1, number_of_plots),
        axes_pad=0.13,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1,
    )    

    for i in range(number_of_plots):
        im = grid[i].quiver(
            X,
            Y,
            Us[i],
            Vs[i],
            norms[i],
            cmap='coolwarm',
            width=0.02,
            scale=25, 
        )             
        grid[i].title.set_text(subtitles[i])
    
    # add color bar
    cbar_pot = grid[i].cax.colorbar(im)
    cbar_pot = grid.cbar_axes[0].colorbar(im)

    # save figure
    fig.subplots_adjust(top=0.8)
    fig.savefig(file_path, format='png', dpi=300,bbox_inches='tight')   


def plot_3well_effcurrent(eff_vectors_unit, colors, xn, yn, background, datashape, extent, timeframe, size,titles):
    # TODO document method
    
    fig = plt.figure(figsize=size)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, timeframe),
                axes_pad=0.13,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )    
    
    #if timeframe == 1:
     #   ax=[ax]
       # if np.isnan(eff_vectors_unit).all()==False: #if not all values are nan
        #    plt.imshow(background.reshape(datashape), cmap='Greys', alpha=.4, ax=ax, origin='lower', extent=extent)
         #   plt.quiver(xn,yn,list(eff_vectors_unit[:,0]),list(eff_vectors_unit[:,1]),colors,cmap='coolwarm', width=0.02, scale=25)    
    i=0
    for ax in grid:
        if np.isnan(eff_vectors_unit[i,:,:]).all()==False: #if not all values are nan
            ax.imshow(background.reshape(datashape), cmap='Greys', alpha=.4, origin='lower', extent=extent)
            im = ax.quiver(xn,yn,list(eff_vectors_unit[i,:,0]),list(eff_vectors_unit[i,:,1]),colors[i],cmap='coolwarm', width=0.02, scale=25)             
            ax.set_title(titles[i])  
        i = i + 1
    #fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True) 
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)
    cbar = grid.cbar_axes[0].colorbar(im)
    return fig


def plot_3well( data,datashape, extent, timeframe, size, v_min, v_max, titles,background=None):
    # TODO document method
    if background is None: 
        background = np.ones(datashape[0]*datashape[1])
        
    fig = plt.figure(figsize=size)

    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, timeframe),
                axes_pad=0.13,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    
 
    #if timeframe == 1:
     #   ax=[ax]
    i=0
    for ax in grid: #i in range(timeframe):
        if np.isnan(data[i,:]).all()==False: #if not all values are nan
            ax.imshow(background.reshape(datashape), cmap='Greys', alpha=1, origin='lower', extent=extent)
            im = ax.imshow(data[i,:].reshape(datashape), vmin=v_min, vmax=v_max, origin='lower', alpha=0.9, extent=extent)
 
            ax.set_title(titles[i])  
        i = i + 1
            
    #fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True) 
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)#%.0e')
    cbar = grid.cbar_axes[0].colorbar(im)
   # ticks = np.linspace(np.min(data),np.max(data), 4) 
    #cb = grid.cbar_axes[0].colorbar(im, ticks=ticks)
    #cb.ax.set_yticklabels(np.array(['1','2','3','4']))
#    cb.ax.text(-0.25, 1, r'$\times$10$^{-1}$', va='bottom', ha='left')
   # fig.colorbar(im, ax=ax[:], shrink=0.95)
 
    return fig
