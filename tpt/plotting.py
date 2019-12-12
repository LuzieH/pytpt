import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx
from mpl_toolkits.axes_grid1 import AxesGrid

VIRIDIS = cm.get_cmap('viridis', 12) 

def plot_network_density(data, graphs, pos, labels, v_min, v_max, file_path, title, subtitles=None):               
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
    width_colorbar = 0.5
    size = (width_plot*num_plots, height_plot)                                                                        
    #size = (width_plot*num_plots + width_colorbar, height_plot)                                                                        
                                                                                                   
    fig, ax = plt.subplots(1, num_plots, sharex='col',                                             
                           sharey='row', figsize=size)                                             
    if num_plots == 1:                                                                             
        ax = [ax]                                                                                  
    for i in range(num_plots):                                                                     
        nx.draw(graphs[i], pos=pos, labels=labels, node_color=data[i],                             
                cmap=VIRIDIS, ax=ax[i], vmin=v_min, vmax=v_max)                                                  
        if subtitles is not None:                                                                  
            ax[i].set_title(subtitles[i])                                                          

    sm = plt.cm.ScalarMappable(
        cmap=VIRIDIS,
        norm=plt.Normalize(vmin=v_min, vmax=v_max)
    )
    #sm._A = []
    #fig.colorbar(
    #    sm,
    #    ax=ax[:],
    #    orientation='vertical',
    #    pad=0.05,
    #    shrink=0.8,
    #)
    fig.suptitle(title)                                                                            
    fig.subplots_adjust(top=0.8)                                                                   
    fig.savefig(file_path, dpi=100)  

    
def plot_network_effective_current(weights, pos, labels, v_min, v_max, file_path, title, subtitles=None):          
    # TODO document method                                                                         
                                                                                                   
    timeframes = len(weights)                                                                      
    size = (2*timeframes, 2)                                                                       
    fig, ax = plt.subplots(1, timeframes, sharex='col',                                            
                           sharey='row', figsize=size)                                             
    if timeframes == 1:                                                                            
        ax = [ax]                                                                                  
    for n in range(timeframes):                                                                    
        if not np.isnan(weights[n]).any():                                                         
            A_eff = (weights[n, :, :] > 0)*1                                                       
            G_eff = nx.DiGraph(A_eff)                                                              
            nbr_edges = int(np.sum(A_eff))                                                         
            edge_colors = np.zeros(nbr_edges)                                                      
            widths = np.zeros(nbr_edges)                                                           
                                                                                                   
            for j in np.arange(nbr_edges):                                                         
                edge_colors[j] = weights[                                                          
                    n,                                                                             
                    np.array(G_eff.edges())[j, 0],                                                 
                    np.array(G_eff.edges())[j, 1],                                                 
                ]                                                                                  
                widths[j] = 150*edge_colors[j]                                                     
                                                                                                   
            nx.draw_networkx_nodes(G_eff, pos, ax=ax[n])                                           
            nx.draw_networkx_edges(                                                                
                G_eff,                                                                             
                pos,                                                                               
                ax=ax[n],                                                                          
                arrowsize=10,                                                                      
                edge_color=edge_colors,                                                            
                width=widths,                                                                      
                edge_cmap=plt.cm.Blues,                                                            
            )                                                                                      
                                                                                                   
            # labels                                                                               
            nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax[n])                           
            #ax = plt.gca()                                                                        
            ax[n].set_axis_off()                                                                   
            if subtitles is not None:                                                              
                ax[n].set_title(subtitles[n])  # , pad=0)                                          
                                                                                                   
    fig.suptitle(title)                                                                            
    fig.subplots_adjust(top=0.8)                                                                   
    fig.savefig(file_path, dpi=100)


def plot_rate(rate, file_path, title, time_av_rate=None):                                          
    # TODO document method                                                                         
    ncol = 2                                                                                       
    timeframes = len(rate[0])                                                                      
    fig, ax = plt.subplots(1, 1, figsize=(2*timeframes, 2))                                        
                                                                                                   
    plt.scatter(                                                                                   
        x=np.arange(timeframes),                                                                   
        y=rate[0, :],                                                                              
        marker='.',
        color='black',
        label='$k^{A->}$',                                                                         
    )                                                                                              
    plt.scatter(                                                                                   
        x=np.arange(timeframes),                                                                   
        y=rate[1, :],                                                                              
        marker='o',
        color='black',
        facecolors='none',
        edgecolors='black',
        label='$k^{->B}$',                                                                         
    )                                                                                              
    if type(time_av_rate) != type(None):                                                           
        ncol = 3                                                                                   
        ax.hlines(                                                                                 
            y=time_av_rate[0],                                                                     
            xmin=np.arange(timeframes)[0],                                                         
            xmax=np.arange(timeframes)[-1],                                                        
            color='black',
            linestyles='dashed',                                                                   
            label='$\hat{k}^{AB}_N$',                                                              
        )                                                                                          
                                                                                                   
    # Hide the right and top spines                                                                
    ax.spines['right'].set_visible(False)                                                          
    ax.spines['top'].set_visible(False)                                                            
                                                                                                   
    # Only show ticks on the left and bottom spines                                                
    ax.yaxis.set_ticks_position('left')                                                            
    ax.xaxis.set_ticks_position('bottom')                                                          
                                                                                                   
    # add title and legend                                                                         
    plt.title(title)                                                                               
    min_rate = np.nanmin([                                                                         
        np.nanmin(rate[0]),                                                                        
        np.nanmin(rate[1]),                                                                        
    ])                                                                                             
    max_rate = np.nanmax([                                                                         
        np.nanmax(rate[0]),                                                                        
        np.nanmax(rate[1]),                                                                        
    ])                                                                                             
    plt.ylim(                                                                                      
        min_rate - (max_rate-min_rate)/4,                                                          
        max_rate + (max_rate-min_rate)/4,                                                          
    )                                                                                              
    plt.xlabel('n')                                                                                
    plt.ylabel('Discrete rate')                                                                    
    plt.legend(ncol=ncol)                                                                          
                                                                                                   
    fig.savefig(file_path, dpi=100)


def plot_reactiveness(reac_norm_factor, file_path, title):
    # TODO document method
    timeframes = len(reac_norm_factor)

    fig, ax = plt.subplots(1, 1, figsize=(2*timeframes, 2))

    plt.scatter(
        np.arange(timeframes),
        reac_norm_factor[:],
        alpha=0.7,
        label='$\sum_{j \in C} \mu_j^{R}(n)$',
    )

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.title(title)
    min_norm_factor = np.nanmin(reac_norm_factor)
    max_norm_factor = np.nanmax(reac_norm_factor)
    plt.ylim(
        min_norm_factor - (max_norm_factor - min_norm_factor)/4,
        max_norm_factor + (max_norm_factor - min_norm_factor)/4,
    )
    #plt.ylim(-0.002, max_norm_factor*(1+1/10))
    plt.xlabel('n')
    plt.legend()

    fig.savefig(file_path, dpi=100)


def plot_convergence(q_f, q_f_conv, q_b, q_b_conv, scale_type, file_path, title):
    # TODO document method
    assert scale_type in ['linear', 'log', 'symlog', 'logit']

    # compute errors 
    q_f_conv_error = np.linalg.norm(q_f_conv - q_f, ord=2, axis=1)                                  
    q_b_conv_error = np.linalg.norm(q_b_conv - q_b, ord=2, axis=1)                                  

    N_max = len(q_f_conv)

    fig, ax = plt.subplots(1, 1, figsize=(2*6, 5))                                                     

    plt.yscale(scale_type) 
    plt.plot(
        np.arange(1, N_max + 1),
        q_f_conv_error,
        color='b',
        alpha=0.5,
        label='$||q^+ - q^+(0)||_2$',
    )
    plt.plot(
        np.arange(1, N_max + 1),
        q_b_conv_error,
        color='r',
        alpha=0.5,
        label='$||q^- - q^-(0)||_2$',
    )

    plt.title(title)                                                                                   
    plt.xlabel('$N$')                                                                                  
    plt.legend(ncol=2)

    # Hide the right and top spines                                                                    
    ax.spines['right'].set_visible(False)                                                              
    ax.spines['top'].set_visible(False)                                                                
                                                                                                       
    # Only show ticks on the left and bottom spines                                                    
    ax.yaxis.set_ticks_position('left')                                                                
    ax.xaxis.set_ticks_position('bottom')                                                              

    fig.savefig(file_path, dpi=100)  


def plot_3well_potential_and_force(potential, vector_field, vector_field_forced,
                                   file_path, title, subtitles=None):
    number_of_plots = 4 
    size = (4*number_of_plots, 3) 
    fig, ax = plt.subplots(
        nrows=1,
        ncols=number_of_plots,
        sharex='col',
        sharey='row',
        figsize=size,
    )
    plt.title(title)                                                                                   

    # compute mesh grid
    delta = 0.01
    x = np.arange(-2.0, 2.0 + delta, delta)
    y = np.arange(-1.0, 2.0 + delta, delta)
    X, Y = np.meshgrid(x, y)
    
    # compute potential on the grid
    potential = potential(X, Y) 

    im = ax[0].imshow(
        potential,
        origin='lower',
        cmap=VIRIDIS,
        extent=[-2, 2, -1, 2],
        vmin=abs(potential).min(),
        vmax=abs(potential).max(),
    )
    ax[0].title.set_text(subtitles[0])
    
    # make grid coarser
    factor = 20
    X, Y = np.meshgrid(x[::factor], y[::factor])
    
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

    ax[1].quiver(
        X,
        Y,
        U_norm,
        V_norm,
        norm,
        cmap=VIRIDIS,
        width=0.02,
        scale=25, 
    )             
    ax[1].title.set_text(subtitles[1])
    ax[2].quiver(
        X,
        Y,
        U_forced_0_norm,
        V_forced_0_norm,
        norm_forced_0,
        cmap=VIRIDIS,
        width=0.02,
        scale=25, 
    )             
    ax[2].title.set_text(subtitles[2])
    ax[3].quiver(
        X,
        Y,
        U_forced_3_norm,
        V_forced_3_norm,
        norm_forced_3,
        cmap=VIRIDIS,
        width=0.02,
        scale=25, 
    )             
    ax[3].title.set_text(subtitles[3])
    
    vmin = min([
        np.min(potential),
        np.min(norm),
        np.min(norm_forced_0),
        np.min(norm_forced_3),
    ])
    vmax = max([
        np.max(potential),
        np.max(norm),
        np.max(norm_forced_0),
        np.max(norm_forced_3),
    ])
    
    # set scalar mappable for the colorbar
    norm_wrt_potential = plt.Normalize(
        vmax=vmax,
        vmin=vmin,
    )
    sm = plt.cm.ScalarMappable(cmap=VIRIDIS, norm=norm_wrt_potential)
    sm._A = []
    
    # add colorbar
    fig.colorbar(
        sm,
        ax=ax[:],
        orientation='vertical',
        pad=0.03,
    )

    fig.savefig(file_path, dpi=100)  





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
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    return fig


def plot_3well( data,datashape, extent, timeframe, size, v_min, v_max, titles):
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
    i=0
    for ax in grid: #i in range(timeframe):
        if np.isnan(data[i,:]).all()==False: #if not all values are nan
            im = ax.imshow(data[i,:].reshape(datashape), vmin=v_min, vmax=v_max, origin='lower', extent=extent)
            ax.set_title(titles[i])  
        i = i + 1
            
    #fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

   # fig.colorbar(im, ax=ax[:], shrink=0.95)
    return fig
