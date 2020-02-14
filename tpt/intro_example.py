import transition_paths as tp
import transition_paths_periodic as tpp
import transition_paths_finite as tpf

from plotting import plot_3well, \
                     plot_3well_effcurrent, \
                     plot_rate, \
                     plot_reactiveness
import numpy as np
import matplotlib.pyplot as plt

import os.path
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import ScalarFormatter
#
#plot current for finite and inf time, small noise
#plot trajectory that takes lower and upper channel
#plot potential underneath
 
##############################################################################
# load data
 
my_path = os.path.abspath(os.path.dirname(__file__))
T = np.load(os.path.join(my_path, 'data/triplewell_T.npy'))
T_small_noise = np.load(os.path.join(my_path, 'data/triplewell_T_small_noise.npy'))
interval = np.load(os.path.join(my_path, 'data/triplewell_interval.npy'))
dx = np.load(os.path.join(my_path, 'data/triplewell_dx.npy'))
ind_A = np.load(os.path.join(my_path, 'data/triplewell_ind_A.npy'))
ind_B = np.load(os.path.join(my_path, 'data/triplewell_ind_B.npy'))
ind_C = np.load(os.path.join(my_path, 'data/triplewell_ind_C.npy'))
traj = np.load(os.path.join(my_path, 'data/triplewell_traj.npy'))
traj = np.load(os.path.join(my_path, 'data/triplewell_traj.npy'))
traj_ab_lower = np.load(os.path.join(my_path, 'data/triplewell_traj_ab_lower.npy'))
traj_ab_upper = np.load(os.path.join(my_path, 'data/triplewell_traj_ab_upper.npy'))
############################################################################
#state space
x = np.arange(interval[0,0],interval[0,1]+dx, dx) #box centers in x and y direction
y = np.arange(interval[1,0],interval[1,1]+dx, dx)
xv, yv = np.meshgrid(x, y)

xdim = np.shape(xv)[0] #discrete dimension in x and y direction
ydim = np.shape(xv)[1]
dim_st = xdim*ydim # dimension of the statespace
xn=np.reshape(xv,(xdim*ydim,1))
yn=np.reshape(yv,(xdim*ydim,1))
grid = np.squeeze(np.array([xn,yn]))

#T_small_noise=T

#############################################################################
# infinite-time ergodic

# instantiate
well3 = tp.transitions_mcs(T_small_noise, ind_A, ind_B, ind_C)
stat_dens = well3.stationary_density()

# compute committor probabilities
[q_f, q_b] = well3.committor()

# therof compute the normalized reactive density
norm_reac_dens = well3.norm_reac_density()

# and reactive currents
[current, eff_current] = well3.reac_current()
rate = well3.transition_rate()  # AB discrete transition rate

###################################
#finite-time

def Tn(n):  
    return T_small_noise#T_m[np.mod(m,M),:,:].squeeze()

N = 20 #time window

# initial density
init_dens_triple = stat_dens

# instantiate
well3_finite = tpf.transitions_finite_time(Tn, N, ind_A, ind_B,  ind_C, init_dens_triple)
 
dens_f = well3_finite.density()
[q_f_f, q_b_f] = well3_finite.committor()
 
# normalized reactive density
reac_norm_factor_f = well3_finite.reac_norm_factor()
norm_reac_dens_f = well3_finite.norm_reac_density()

# and reactive currents
[current_f, eff_current_f] = well3_finite.reac_current()

[rate_f, time_av_rate_f] = well3_finite.transition_rate()

################################

#define AB sets
densAB = np.zeros(dim_st)
densAB[ind_A]=1
densAB[ind_B]=1

#calculation the effective vector for each state
eff_vectors = np.zeros((dim_st, 2))
eff_vectors_unit = np.zeros((dim_st, 2))
colors = np.zeros(dim_st)
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
            eff_vectors[i,0] += eff_current[i,j] *  (xn[j] - xn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
            eff_vectors[i,1] += eff_current[i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
    colors[i] = np.linalg.norm(eff_vectors[i,:])
    if colors[i]>0:
        eff_vectors_unit[i,:] = eff_vectors[i,:]/colors[i] 
            
#fig = plot_3well_effcurrent(np.array([eff_vectors_unit]), np.array([colors]), xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), 1, (3*1,3),['$f^+$'])
#fig.savefig(os.path.join(charts_path, 'triplewell_eff_intro.eps'), dpi=100,bbox_inches='tight')
#


#######################################

n=int(N/2)
#calculation the effective vector for each state
eff_vectors_f = np.zeros((dim_st, 2))
eff_vectors_unit_f = np.zeros((dim_st, 2))
colors_f = np.zeros(dim_st)
for i in np.arange(dim_st):
    for j in np.arange(dim_st):
        if np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))>0:
            eff_vectors_f[i,0] += eff_current_f[n,i,j] *  (xn[j] - xn[i])*(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]])))   
            eff_vectors_f[i,1] += eff_current_f[n,i,j] *  (yn[j] - yn[i])  *(1/np.linalg.norm(np.array([xn[j] - xn[i],yn[j] - yn[i]]))) 
    colors_f[i] = np.linalg.norm(eff_vectors_f[i,:])
    if colors_f[i]>0:
        eff_vectors_unit_f[i,:] = eff_vectors_f[i,:]/colors_f[i] 
    
   
#fig = plot_3well_effcurrent(np.array([eff_vectors_unit_f]), np.array([colors_f]), xn, yn, densAB,(xdim,ydim), (interval[0,0],interval[0,1],interval[1,0],interval[1,1]), 1, (3*1,3),['$f^+$('+str(n)+'), N = '+str(N)])
#fig.savefig(os.path.join(charts_path, 'triplewell_eff_intro_finite.eps'), dpi=100,bbox_inches='tight')
#



#######################################
 
size = (4,4)
trajec = [np.zeros((2,2)), traj_ab_upper, np.zeros((2,2)), traj_ab_lower] #traj_ab_upper
colorsar = [colors,colors,colors_f,colors_f]
data = [eff_vectors_unit,eff_vectors_unit,eff_vectors_unit_f,eff_vectors_unit_f]
title = np.array(['$f^+$','$f^+$', '$f^+$('+str(n)+'), N = '+str(N),'$f^+$('+str(n)+'), N = '+str(N)])
save = np.array(['eff','eff_traj','eff_f','eff_f_traj'])
 

for k in np.arange(4):
    
    fig = plt.figure(figsize=size)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 1),
                axes_pad=0.13,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )    
    
    for ax in grid:

        im = ax.quiver(xn,yn,list(data[k][:,0]),list(data[k][:,1]),colorsar[k],cmap='inferno_r', width=0.02, scale=25 )   
        ax.imshow(densAB.reshape((xdim,ydim)), cmap='Greys', alpha=.3, origin='lower', extent=(interval[0,0],interval[0,1],interval[1,0],interval[1,1]))
        ax.plot(trajec[k][:,0], trajec[k][:,1],'r', alpha=0.3, linewidth=0.5)
        ax.set_title(title[k])  
 
    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True) 
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)
    cbar = grid.cbar_axes[0].colorbar(im)
    
    fig.savefig(os.path.join(charts_path, save[k]+'_intro.eps'), dpi=150,bbox_inches='tight')
    
######################################## V0
#state space
dx2 = 2./(10**2)
interval2 = interval #np.array([[-2.2,2.2],[-1.5,2.5]]) 
x2 = np.arange(interval2[0,0],interval2[0,1]+dx2, dx2) #box centers in x and y direction
y2 = np.arange(interval2[1,0],interval2[1,1]+dx2, dx2)
xv2, yv2 = np.meshgrid(x2, y2)

xdim2 = np.shape(xv2)[0] #discrete dimension in x and y direction
ydim2 = np.shape(xv2)[1]
dim_st2 = xdim2*ydim2 # dimension of the statespace
xn2=np.reshape(xv2,(xdim2*ydim2,1))
yn2=np.reshape(yv2,(xdim2*ydim2,1))
pot = V0(xn2, yn2)

data = np.array([pot])
v_min = np.nanmin(data)
v_max = np.nanmax(data)
fig = plot_3well(data, (xdim2,ydim2), (interval2[0,0],interval2[0,1],interval2[1,0],interval2[1,1]) , 1, (3*1,3), v_min, v_max, ['$V(x,y)$'])
#fig.savefig(os.path.join(charts_path, 'V_intro.eps'), dpi=100,bbox_inches='tight')

#plt.imshow(pot.reshape((xdim2,ydim2)), origin='lower', extent=(interval2[0,0],interval2[0,1],interval2[1,0],interval2[1,1])) 
######################trajectory
#
#dt=0.02
#steps=50000
#sigma_small = 0.5
#limits_x = interval[0,:]
#limits_y = interval[1,:]
#traj2 = sample_stat_trajectories_2D(dV0, sigma_small, dt, steps, limits_x, limits_y)
#plt.imshow(pot.reshape((xdim2,ydim2)), origin='lower', extent=(interval2[0,0],interval2[0,1],interval2[1,0],interval2[1,1])); plt.plot(traj2[:,0],traj2[:,1],'w', alpha=0.3, linewidth=0.5)
##
#setABC = np.zeros((steps,3))
#for i in np.arange(steps):
#    setABC[i,:] = np.array([set_A_triplewell(traj2[i,:], A_center, radius_setAB), set_B_triplewell(traj2[i,:],B_center, radius_setAB),set_C_triplewell(traj2[i,:],A_center, B_center, radius_setAB)])
#    
#listAout = []
#listBin = []
##listAin = []
##listBout = []
#lastABset = 1
#AB_trans = []
#for i in np.arange(steps-1):
#    if setABC[i,2]==1 and setABC[i+1,1]==1:
#        listBin.append(i)
#        if lastABset==0:
#            AB_trans.append(np.array([listAout[-1],listBin[-1]]))    
#    if setABC[i,0]==1 and setABC[i+1,2]==1:
#        listAout.append(i)
#        lastABset = 0
#    if setABC[i,1]==1 and setABC[i+1,2]==1:
#        #listBout.append(i)
#        lastABset=1
##    if setABC[i,2]==1 and setABC[i+1,0]==1
##        listAin.append(i)  
#        
##traj_ab_upper = 
##traj_ab_lower = traj[1600:1660,:]
#


#####################plotting
trajec = [np.zeros((2,2)), traj_ab_upper, traj_ab_lower] #traj_ab_upper
title = np.array(['$V(x,y)$','$V(x,y)$','$V(x,y)$'])
save = np.array([ 'V', 'trajupper', 'trajlower'])
 

for k in np.arange(3):
    
    fig = plt.figure(figsize=size)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 1),
                    axes_pad=0.13,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
 
    for ax in grid: #i in range(timeframe):
        im = ax.imshow(pot.reshape((xdim2,ydim2)), origin='lower', cmap='inferno_r', extent=(interval2[0,0],interval2[0,1],interval2[1,0],interval2[1,1]))
        ax.imshow(densAB.reshape((xdim,ydim)), cmap='Greys', alpha=.15, origin='lower', extent=(interval[0,0],interval[0,1],interval[1,0],interval[1,1]))
        ax.plot(trajec[k][:,0], trajec[k][:,1],'k', alpha=0.7, linewidth=0.5)
        ax.set_title(title[k])  
 
            
    #fig.suptitle(title)
    fig.subplots_adjust(top=0.8)
    sfmt=ScalarFormatter(useMathText=True) 
    sfmt.set_powerlimits((0, 0))
    cbar = ax.cax.colorbar(im, format=sfmt)#%.0e')
    cbar = grid.cbar_axes[0].colorbar(im)
    
    fig.savefig(os.path.join(charts_path, save[k]+'_intro.eps'), dpi=150,bbox_inches='tight')

#####################plotting
    
fig = plt.figure(figsize=size)
grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 1),
                axes_pad=0.13,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
 
for ax in grid: #i in range(timeframe):
    im = ax.imshow(pot.reshape((xdim2,ydim2)), cmap='inferno_r', origin='lower', extent=(interval2[0,0],interval2[0,1],interval2[1,0],interval2[1,1]))
    ax.imshow(densAB.reshape((xdim,ydim)), cmap='Greys', alpha=.15, origin='lower', extent=(interval[0,0],interval[0,1],interval[1,0],interval[1,1]))
    ax.plot(traj_ab_upper[:,0], traj_ab_upper[:,1],'k', alpha=0.7, linewidth=0.5)
    ax.plot(traj_ab_lower[:,0], traj_ab_lower[:,1],'r', alpha=1, linewidth=0.5)
    ax.set_title('$V(x,y)$')  
 
        
#fig.suptitle(title)
fig.subplots_adjust(top=0.8)
sfmt=ScalarFormatter(useMathText=True) 
sfmt.set_powerlimits((0, 0))
cbar = ax.cax.colorbar(im, format=sfmt)#%.0e')
cbar = grid.cbar_axes[0].colorbar(im)

fig.savefig(os.path.join(charts_path, 'trajlower_upper'+'_intro.eps'), dpi=150,bbox_inches='tight')
