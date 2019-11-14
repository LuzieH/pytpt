import numpy as np
import matplotlib.pyplot as plt


 
def sample_stat_trajectories_2D(force_function, sigma, dt, steps, limits_x, limits_y):
    '''returns a sampled trajectory of length steps and with an initial position
    drawn uniformly from limits_x, limits_y. The trajectory is sampled from the 
    diffusion process: dX = -F(X) dt + sigma dW '''
    traj = np.zeros((steps, 2))
    traj[0,:] = np.array([np.random.uniform(limits_x[0],limits_x[1]),np.random.uniform(limits_y[0],limits_y[1])])
    
    for i in np.arange(1,steps):
        traj[i,:] = traj[i-1,:] - force_function(traj[i-1,0],traj[i-1,1])*dt + np.sqrt(dt)*sigma*np.random.randn(2)
    return traj


def transitionmatrix_2D(force,  sigma, dt, lag, Nstep, interval, dx_power, x, y, dim):
    ''' This function returns a row-stochastic transition matrix by counting transitions of the
    overdamped langevin process with dV=force. The state space is discretized into boxes
    and jumps from each box are counted. 
    '''

    x=np.round(x,dx_power)
    y=np.round(y,dx_power)
    
    xy=[x,y]
    xv, yv = np.meshgrid(x, y)
    
    xdim = np.shape(xv)[1]
    ydim = np.shape(xv)[0]
        
    xn=np.reshape(xv,(xdim*ydim,1))
    yn=np.reshape(yv,(xdim*ydim,1))
    
    grid = np.squeeze(np.array([xn,yn]))
    states_dim = np.shape(grid)[1]
    count = np.zeros((states_dim,states_dim))
    to_box_int=np.zeros(dim)
    
    for j in range(Nstep):
            for seed in range(states_dim):
                #todo: if cells are large, need to reweigh with stationary density in each cell
                current_X = grid[:,seed]
                for l in range(lag):
                    new_X = current_X - (force(current_X[0],current_X[1]))*dt + sigma*np.sqrt(dt)*np.random.randn(2)
                    for d in range(dim):
                        if new_X[d] <interval[d,0]: #reflective boundary conditions
                            new_X[d] = interval[d,0] + (interval[d,0]-new_X[d])
                        if new_X[d] >interval[d,1]:
                            new_X[d] = interval[d,1] - (new_X[d] - interval[d,1])
                    current_X=new_X
                from_box = seed
                for d in range(dim):
                    to_box_int[d]=np.where(xy[d]==np.round(new_X[d],dx_power))[0][0]
                to_box = np.int(to_box_int[0] + xdim*to_box_int[1])
                count[from_box,to_box] += 1.
    return count/Nstep

#if the dynamics are periodic or time-dependent, need to get transition matrices
# for each of the forces
