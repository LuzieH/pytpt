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


def plot_convergence(N_ex, q_f, q_f_conv, file_path, title):
    # TODO document method

    fig, ax = plt.subplots(1, 1, figsize=(2*6, 5))                                                     
    convergence_error = np.linalg.norm(q_f_conv - q_f, ord=2, axis=1)                                  
    plt.plot(np.arange(1, N_ex), convergence_error)  # , s=5, marker='o')                              
    plt.title(title)                                                                                   
    plt.xlabel('$N$')                                                                                  
    plt.ylabel('$l_2$-Error $||q^+ - q^+(0)||$ ')                                                      
    # Hide the right and top spines                                                                    
    ax.spines['right'].set_visible(False)                                                              
    ax.spines['top'].set_visible(False)                                                                
                                                                                                       
    # Only show ticks on the left and bottom spines                                                    
    ax.yaxis.set_ticks_position('left')                                                                
    ax.xaxis.set_ticks_position('bottom')                                                              
    fig.savefig(file_path, dpi=100)  
