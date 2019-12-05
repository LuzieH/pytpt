import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


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
    size = (2*num_plots, 2)                                                                        
                                                                                                   
    fig, ax = plt.subplots(1, num_plots, sharex='col',                                             
                           sharey='row', figsize=size)                                             
    if num_plots == 1:                                                                             
        ax = [ax]                                                                                  
    for i in range(num_plots):                                                                     
        nx.draw(graphs[i], pos=pos, labels=labels, node_color=data[i],                             
                ax=ax[i], vmin=v_min, vmax=v_max)                                                  
        if subtitles is not None:                                                                  
            ax[i].set_title(subtitles[i])                                                          
                                                                                                   
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
        alpha=0.7,                                                                                 
        label='$k^{A->}$',                                                                         
    )                                                                                              
    plt.scatter(                                                                                   
        x=np.arange(timeframes),                                                                   
        y=rate[1, :],                                                                              
        alpha=0.7,                                                                                 
        label='$k^{->B}$',                                                                         
    )                                                                                              
    if type(time_av_rate) != type(None):                                                           
        ncol = 3                                                                                   
        ax.hlines(                                                                                 
            y=time_av_rate[0],                                                                     
            xmin=np.arange(timeframes)[0],                                                         
            xmax=np.arange(timeframes)[-1],                                                        
            color='r',                                                                             
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


def plot_convergence(q_f, q_f_conv, q_b, q_b_conv, file_path, title):
    # TODO document method

    # compute errors 
    q_f_conv_error = np.linalg.norm(q_f_conv - q_f, ord=2, axis=1)                                  
    q_b_conv_error = np.linalg.norm(q_b_conv - q_b, ord=2, axis=1)                                  

    N_max = len(q_f_conv)

    fig, ax = plt.subplots(1, 1, figsize=(2*6, 5))                                                     

#    plt.scatter(                                                                                   
#        x=np.arange(1, N_max +1),                                                                   
#        y=q_f_conv_error,                                                                              
#        #color=blue,
#        alpha=0.7,                                                                                 
#        label='$l_2$-Error $||q^+ - q^+(0)||$',
#    )                                                                                              
#    plt.scatter(                                                                                   
#        x=np.arange(1, N_max +1),                                                                   
#        y=q_b_conv_error,                                                                              
#        #color=red,
#        alpha=0.7,                                                                                 
#        label='$l_2$-Error $||q^- - q^-(0)||$',
#    )                                                                                              
#    plt.scatter(                                                                                   
#        x=np.arange(1, N_max +1),                                                                   
#        y=dens_conv_error,                                                                              
#        #color=red,
#        alpha=0.7,                                                                                 
#        label='$l_2$-Error $||\pi - \pi_0||$',
#    )                                                                                              
    plt.yscale('log') 
    plt.plot(np.arange(1, N_max + 1), q_f_conv_error)  # , s=5, marker='o')                              
    plt.plot(np.arange(1, N_max + 1), q_b_conv_error)  # , s=5, marker='o')                              

    plt.title(title)                                                                                   
    plt.xlabel('$N$')                                                                                  
    #plt.ylabel('$l_2$-Error $||q^+ - q^+(0)||$ ')                                                      
    #plt.legend(ncol=3)

    # Hide the right and top spines                                                                    
    ax.spines['right'].set_visible(False)                                                              
    ax.spines['top'].set_visible(False)                                                                
                                                                                                       
    # Only show ticks on the left and bottom spines                                                    
    ax.yaxis.set_ticks_position('left')                                                                
    ax.xaxis.set_ticks_position('bottom')                                                              

    fig.savefig(file_path, dpi=100)  


def plot_3well_effcurrent(eff_vectors_unit, colors, xn, yn, background, datashape, extent, timeframe, size,title, subtitles=None):
    # TODO document method
    
    fig, ax = plt.subplots(1, timeframe, sharex='col',
                           sharey='row', figsize=size)
    if timeframe == 1:
        ax=[ax]
       # if np.isnan(eff_vectors_unit).all()==False: #if not all values are nan
        #    plt.imshow(background.reshape(datashape), cmap='Greys', alpha=.4, ax=ax, origin='lower', extent=extent)
         #   plt.quiver(xn,yn,list(eff_vectors_unit[:,0]),list(eff_vectors_unit[:,1]),colors,cmap='coolwarm', width=0.02, scale=25)    
 
    for i in range(timeframe):
        if np.isnan(eff_vectors_unit[i,:,:]).all()==False: #if not all values are nan
            ax[i].imshow(background.reshape(datashape), cmap='Greys', alpha=.4, origin='lower', extent=extent)
            ax[i].quiver(xn,yn,list(eff_vectors_unit[i,:,0]),list(eff_vectors_unit[i,:,1]),colors[i],cmap='coolwarm', width=0.02, scale=25)             
        if subtitles is not None:
            ax[i].set_title(subtitles[i])  
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    return fig



def plot_3well( data,datashape, extent, timeframe, size, v_min, v_max, title, subtitles=None):
    # TODO document method
    
    fig, ax = plt.subplots(1, timeframe, sharex='col',
                           sharey='row', figsize=size)
    if timeframe == 1:
        ax=[ax]

    for i in range(timeframe):
        if np.isnan(data[i,:]).all()==False: #if not all values are nan
            ax[i].imshow(data[i,:].reshape(datashape), vmin=v_min, vmax=v_max, origin='lower', extent=extent)
        if subtitles is not None:
            ax[i].set_title(subtitles[i])  
            
    fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    return fig
