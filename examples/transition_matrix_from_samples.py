import numpy as np
 
 
def sample_stat_trajectories_2D(force_function, sigma, dt, steps,\
                                limits_x, limits_y):
    '''returns a sampled trajectory of length steps and with an initial
    position drawn uniformly from limits_x, limits_y. The trajectory is 
    sampled from the diffusion process: dX = F(X) dt + sigma dW '''
    traj = np.zeros((steps, 2))
    traj[0, :] = np.array([
        np.random.uniform(limits_x[0], limits_x[1]), \
        np.random.uniform(limits_y[0], limits_y[1])
    ])
    
    for i in np.arange(1, steps):
        traj[i, :] = traj[i-1, :] + force_function(traj[i-1, 0], traj[i-1, 1])*dt \
        + np.sqrt(dt)*sigma*np.random.randn(2)
    return traj


def transitionmatrix_2D(force, sigma, dt, lag, Nstep, interval, x, y, \
                        dx, dim):
    ''' This function returns a row-stochastic transition matrix by 
    counting transitions of the overdamped langevin process 
    dX = F(X) dt + sigma dW, e.g. with F = -dV. The state space is 
    discretized into boxes and jumps from each box are counted. 
    
    Literature on count matrices: Frank Noe, 
    publications.imp.fu-berlin.de/1699/1/autocorrelation_counts.pdf
    '''
 
    xy = [x,y]
    xv, yv = np.meshgrid(x, y)
    
    xdim = np.shape(xv)[1]
    ydim = np.shape(xv)[0]
        
    xn = np.reshape(xv, (xdim*ydim, 1))
    yn = np.reshape(yv, (xdim*ydim, 1))
    
    grid = np.squeeze(np.array([xn, yn]))
    states_dim = np.shape(grid)[1]
    count = np.zeros((states_dim, states_dim))
    to_box_int = np.zeros(dim)
    
    for j in range(Nstep):
            for seed in range(states_dim):
                # draw uniform sample from grid cell
                current_X = grid[:, seed] + np.random.uniform(-1, 1, size=2)*dx
                for l in range(lag):
                    new_X = current_X + (force(current_X[0],current_X[1]))*dt \
                    + sigma*np.sqrt(dt)*np.random.randn(2)
                    for d in range(dim): #dim=2
                        #reflective boundary conditions
                        if new_X[d] <interval[d,0]: 
                            new_X[d] = interval[d,0] \
                            + (interval[d,0]-new_X[d])
                        if new_X[d] >interval[d,1]:
                            new_X[d] = interval[d,1] \
                            - (new_X[d] - interval[d,1])
                    current_X=new_X
                from_box = seed
                for d in range(dim):

                    to_box_int[d]=np.argmin(np.abs(new_X[d]-xy[d]))
                to_box = np.int(to_box_int[0] + xdim*to_box_int[1])
                count[from_box,to_box] += 1.
    return count/Nstep