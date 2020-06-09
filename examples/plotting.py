import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import ScalarFormatter

PLASMA = mpl.cm.get_cmap('plasma_r', 500)
NETWORK_CMAP = mpl.colors.ListedColormap(
    PLASMA(np.linspace(0.10, 0.65, 275))
)
TRIPLEWELL_CMAP = mpl.cm.get_cmap('inferno_r', 512)
#INFERNO_REV = mpl.cm.get_cmap('inferno_r', 512)
#TRIPLEWELL_CMAP = mpl.colors.ListedColormap(
#INFERNO_REV(np.linspace(0.05, 0.925, 448)))


def plot_colorbar_only(file_path):
    '''
    Plots the colorbar and saves it under file_path.
    '''
    # https://matplotlib.org/examples/api/colorbar_only.html

    fig, ax = plt.subplots(figsize=(0.15, 2))

    cmap = NETWORK_CMAP
    # norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)

    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        orientation='vertical',
    )
    # remove ticks and values
    cb.set_ticks([])

    # ticks size and label size
    # cb1.ax.tick_params(length=4, width=0.5, labelsize=8)

    fig.subplots_adjust(hspace=0.1)
    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_network_density(data, graphs, pos, labels, v_min, v_max,
                         file_path, title=None, subtitles=None):
    """
    For a Markov chain on a network of nodes this function plots
    in several subplots several densities (e.g. for different
    times in a period/time interval), the values of the densities
    are indicated by the node colors.

    Args:
    data : ndarray of size (# subplots/time instances, # nodes of network)
        array of densities for each subplot
    graphs : list
        list of networkx graphs for the different subplots/time instances
    pos : dict
        positions of node for each state
    labels : dict
        labels of the different states
    vmin : float
        minimum value of the colorbar
    vmax : float
        maximum value of the colorbar
    title : string
        overall title
    subtitles : list of strings
        subtitles for the subplots
    file_path: string
        path to where the file should be saved eg ".../plots/image.png""
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
        nx.draw(graphs[i], pos=pos, labels=labels, node_color=data[i],\
                node_size=500, cmap=NETWORK_CMAP, ax=ax[i], \
                vmin=v_min, vmax=v_max, alpha=0.8, font_size=12)
        if subtitles is not None:
            ax[i].set_title(subtitles[i], fontsize=14)

    if title is not None:
        fig.suptitle(title)

    fig.subplots_adjust(
        top=0.8,
        hspace=0.1,
    )

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_network_effective_current(eff_current, pos, labels, v_min, v_max,
                                   file_path, title=None, subtitles=None):
    """
    For a Markov chain on a network of nodes this function plots
    in several subplots several currents, the value of the current between
    two nodes is indicated by color and thickness of the edge.

    Args:
    eff_current : ndarray of size (# subplots, # nodes,  # nodes)
        array of currents between states/nodes for each subplot/time instance
    pos : dict
        positions of node for each state
    labels : dict
        labels of the different states
    vmin : float
        minimum value of the colorbar
    vmax : float
        maximum value of the colorbar
    file_path: string
        path to where the file should be saved eg ".../plots/image.png""
    subtitles : list of strings
        subtitles for the subplots
    title : string
        overall title
    """

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

    fig.savefig(file_path, format='png', dpi=200, bbox_inches='tight')


def plot_network_effcurrent_and_rate(eff_current, shifted_rate, pos, labels, v_min,
                                     v_max, file_path, title=None, subtitles=None):
    """
    For a Markov chain on a network of nodes this function plots
    in several subplots several currents, the value of the current between
    two nodes is indicated by color and thickness of the edge, also the
    outrate of A and inrate into B are shown in the corresponding nodes.

    Args:
    eff_current : ndarray of size (# subplots x # nodes # nodes)
        array of currents for each subplot (e.g. different time instances)
    shifted_rate: ndarray of size (# subplots, 2)
        for each subplot the out-of-A and into-B rate but with
        shifted time indices to agree with the current's time
    pos : dict
        positions of node for each state
    labels : dict
        labels of the different states
    vmin : float
        minimum value of the colorbar
    vmax : float
        maximum value of the colorbar
    file_path: string
        path to where the file should be saved eg ".../plots/image.png""
    subtitles : list of strings
        subtitles for the subplots
    title : string
        overall title
    """

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
                node_size=500,
                alpha=0.8,
                ax=ax[n],
                font_size=12,
            )
            nx.draw_networkx_nodes(
                G_eff,
                pos,
                nodelist=[0, 4],
                node_color=shifted_rate[n],
                node_size=500,
                cmap=NETWORK_CMAP,
                alpha=0.8,
                ax=ax[n],
                font_size=12,
            )

            # edges
            nbr_edges = len(G_eff.edges())
            edge_eff_current = np.zeros(nbr_edges)
            # widths = np.zeros(nbr_edges)
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

            # labels
            nx.draw_networkx_labels(G_eff, pos, labels=labels, ax=ax[n])
            ax[n].set_axis_off()

            if subtitles is not None:
                ax[n].set_title(subtitles[n], fontsize=14)

    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(
        top=0.8,
        hspace=0.1,
    )

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_rate(rate, file_path, title, xlabel, average_rate_legend='$\hat{k}^{AB}$',
              time_av_rate=None):
    """
    This function plots/saves the out-of-A and into-B rate in time.

    Args:
    rate : ndarray of size (# times, 2)
        out and in rate for each time point (during a period or finite
        time interval)
    file_path: string
        path to where the file should be saved eg ".../plots/image.png""
    title : string
        overall title
    xlabel : string
        label of x axis
    average_rate_legend : string
    time_av_rate : float
        time-averaged rate
    """
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
            label=average_rate_legend,
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

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_reactiveness(reac_norm_fact, file_path, title):
    """
    This function plots the probability to be reactive (given by
    reac_norm_fact) in time.

    Args:
    reac_norm_fact : ndarray of size (# times, 1)
        reac_norm_fact for each time point
    file_path: string
        path to where the file should be saved eg ".../plots/image.png""
    title : string
        overall title
    """
    timeframes = len(reac_norm_fact)

    fig, ax = plt.subplots(1, 1, figsize=(5*timeframes, 5))

    plt.scatter(
        np.arange(timeframes),
        reac_norm_fact[:],
        color='black',
        s=20,
        label='$\sum_{j \in C} \mu_j^{R}(n)$',
    )

    plt.title(title, fontsize=20)
    min_norm_factor = np.nanmin(reac_norm_fact)
    max_norm_factor = np.nanmax(reac_norm_fact)
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

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_convergence(q_f, q_f_conv, q_b, q_b_conv, scale_type, file_path, title):
    '''
    Plots (and saves the plot) the convergence of the forward and
    backward committor for a finite-time, stationary system to the forward
    and backward committor of an infinite-time, stationary system,
    when the time interval becomes very large.

    Args:
    q_f : ndarray of the size (# states)
        forward committor of the inf-time, stationary system
    q_f_conv : ndarray of the size (# of time interval lenghts, # states)
        forward committor at a middle time point of the finite-time interval
        for different time-interval lengths
    q_b : ndarray of the size (# states)
        backward committor of the inf-time, stationary system
    q_b_conv : ndarray of the size (# of time interval lenghts, # states)
        backward committor at a middle time point of the finite-time interval
        for different time-interval lengths
    scale_type : 'linear', 'log', 'symlog' or 'logit'
        y axis scaling
    file_path : string
        path to where the file should be saved eg ".../plots/image.png""
    title : string

    '''
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

    fig.savefig(file_path, format='png', dpi=300, bbox_inches='tight')


def plot_3well_potential(potential,  title, file_path=None, subtitles=None):
    '''
    Plots (and saves the plot) the triplewell potential.

    Args:
    potential : function of x and y
        defines the potential to be plotted in terms of x and y coordinates
    file_path : string or None
        path to where the file should be saved eg ".../plots/image.png""
    title : string
    subtitles : string

    '''
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
            cmap=TRIPLEWELL_CMAP,
            extent=[-2, 2, -1, 2],
        )
        grid[i].title.set_text(subtitles[i])

    # add color bar
    cbar_pot = grid[i].cax.colorbar(im)
    cbar_pot = grid.cbar_axes[0].colorbar(im)

    # save figure
    fig.subplots_adjust(top=0.8)

    if file_path is not None:
        fig.savefig(file_path, format='png', dpi=300)

def plot_3well_vector_field(vector_field, vector_field_forced,
                             title, file_path = None, subtitles=None):
    '''
    Plots (and saves the plot) the vector field -grad V of the potential V,
    as well as two time instances (m=0, 3) of the forced vector field.

    Args:
    vector_field : function of x and y
        defines the vector field
    vector_field_forced : function of x, y and time
        defines the forced vector field at discrete times
    file_path : string or None
        path to where the file should be saved eg ".../plots/image.png""
    title : string
    subtitles : string

    '''
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
            cmap=TRIPLEWELL_CMAP,
            width=0.02,
            scale=25,
        )
        grid[i].title.set_text(subtitles[i])

    # add color bar
    cbar_pot = grid[i].cax.colorbar(im)
    cbar_pot = grid.cbar_axes[0].colorbar(im)

    # save figure
    fig.subplots_adjust(top=0.8)

    if file_path is not None:
        fig.savefig(file_path, format='png', dpi=300)


def plot_3well(data, datashape, extent, timeframe, size, v_min, v_max,
               titles, file_path=None, background=None):
    """
    For a Markov chain on discrete 2D statespace this function plots
    in several subplots (e.g. for several time points) several densities.

    Args:
    data : ndarray of size (# subplots, # states)
        array of densities for each subplot/time point
    datashape : (xdim, ydim)
        dimension (int) of statespace in x and y direction
    extent : (x_min, x_max, y_min, y_max)
        gives the limits of the statespace
    timeframe : int
        number of subplots/time frames
    size: (xsize, ysize)
        figure size in x and y direction
    v_min : float
        minimum value of the colorbar
    v_max : float
        maximum value of the colorbar
    titles : list of strings
        titles for the different subplots
    background: ndarray of size # states
        if given, is plotted in the background and the foreground is slightly
        transparent
    file_path: string or None
        path to where the file should be saved eg ".../plots/image.png""
    """
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

    i=0
    for ax in grid:
        if np.isnan(data[i,:]).all()==False: #if not all values are nan
            ax.imshow(background.reshape(datashape), cmap='Greys', alpha=1, \
                     origin='lower', extent=extent)
            im = ax.imshow(
                data[i,:].reshape(datashape),
                vmin=v_min,
                vmax=v_max,
                cmap=TRIPLEWELL_CMAP,
                origin='lower',
                alpha=0.9,
                extent=extent,
            )

            ax.set_title(titles[i])
        i = i + 1

    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)
    cbar = grid.cbar_axes[0].colorbar(im)

    if file_path is not None:
        fig.savefig(file_path, format='png', dpi=100, bbox_inches='tight')

def plot_3well_effcurrent(eff_vectors_unit, colors, xn, yn, background,datashape,
                          extent, timeframe, size, titles, file_path=None):
    """
    For a Markov chain on discrete 2D statespace this function plots
    in several subplots several vectorfields/effective currents.

    Args:
    eff_vectors_unit : ndarray of size (# subplots, # states, 2)
        array of normalized 2D vectors (effective currents) attached to
        each state and given for each subplot
    colors: ndarray of size (#subplots, # states)
        the length of the not normalized vectors
    xn: ndarray
        x values of all state centers
    yn: ndarray
        y values of all state centers
    background: ndarray of size # states
        if given, is plotted in the background and the foreground is slightly
        transparent
    datashape : (xdim, ydim)
        dimension (int) of statespace in x and y direction
    extent : (x_min, x_max, y_min, y_max)
        gives the limits of the statespace
    timeframe : int
        number of subplots/time frames
    size: (xsize, ysize)
        figure size in x and y direction
        maximum value of the colorbar
    titles : list of strings
        titles for the different subplots
    file_path: string or None
        path to where the file should be saved eg ".../plots/image.png""
    """

    fig = plt.figure(figsize=size)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, timeframe),
                axes_pad=0.13,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )

    i=0
    for ax in grid:
        #if not all values are nan
        if np.isnan(eff_vectors_unit[i,:,:]).all()==False:
            ax.imshow(background.reshape(datashape), alpha=1, vmin=0, vmax=3,\
                      cmap='Greys', origin='lower', extent=extent)
            im = ax.quiver(xn,yn,list(eff_vectors_unit[i,:,0]),\
                           list(eff_vectors_unit[i,:,1]),colors[i],\
                           cmap=TRIPLEWELL_CMAP, width=0.02, scale=25)
            ax.set_title(titles[i])
        i = i + 1

    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)
    cbar = grid.cbar_axes[0].colorbar(im)

    if file_path is not None:
        fig.savefig(file_path, format='png', dpi=100, bbox_inches='tight')

