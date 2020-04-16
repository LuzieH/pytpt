import numpy as np


def traj_2D(force_function, sigma, dt, steps,\
                                limits_x, limits_y):
    '''returns a sampled trajectory with an initial position drawn 
    uniformly from limits_x, limits_y. The trajectory is 
    sampled from the diffusion process: dX = F(X) dt + sigma dW by an 
    Euler-Maruyama discretization.
    
    Args:
        force_function: function of x and y
            the forcing function F(x,y)=- (Grad V)(x,y) that defines the 
            diffusion process
        sigma: float>0
            noise strength
        dt: float
            discrete time step size
        steps: int
            number of sampled dt time steps
        limits_x: list
            gives the boundary of the box in x direction
        limits_y: list
            gives the boundary of the box in y direction        
    '''
    traj = np.zeros((steps, 2))
    traj[0, :] = np.array([
        np.random.uniform(limits_x[0], limits_x[1]), \
        np.random.uniform(limits_y[0], limits_y[1])
    ])

    for i in np.arange(1, steps):
        traj[i, :] = traj[i-1, :] + force_function(traj[i-1, 0], traj[i-1, 1])*dt \
        + np.sqrt(dt)*sigma*np.random.randn(2)
    return traj

def transitionmatrix_2D(force, sigma, dt, lag, Nstep, interval, x, y, dx):
    ''' This function returns a row-stochastic transition matrix giving 
    the transition probabilities between spatial grid cells of a 
    discretized state space. The dynamics are given by the overdamped 
    langevin process dX = F(X) dt + sigma dW (e.g. with F = -dV). 
    The transition matrix is estimated as follows: 
        - first trajectory snippets (X_t, X_t+1) are sampled using an 
        Euler-Maruyama discretization of the process
        - then a count matrix between different boxes is constructed and
        normalized to give a row-stochastic matrix. 
    
    Literature on count matrices: Frank Noe, 
    publications.imp.fu-berlin.de/1699/1/autocorrelation_counts.pdf
    
    Args:
        force: function of x and y
            the forcing function F(x,y)=- (Grad V)(x,y) that defines the 
            diffusion process
        sigma: float
            noise strength
        dt: float
            size of time steps used for Euler-Maryama discretization
        lag: int
            the transition matrix maps forward lag*dt in time
        Nstep: int
            number of samples per box
        interval: array of size 2x2
           box limits
        x: array
            box centers in x direction
        y: array
            box centers in y direction
        dx: float
            space discretization
 
        
    '''
 
    xy = [x,y]
    xv, yv = np.meshgrid(x, y)
    
    xdim = np.shape(xv)[1]
    ydim = np.shape(xv)[0]
       
    dim = 2 # dimension of state space
    
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
                    for d in range(dim): # dim=2
                        # reflective boundary conditions
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
