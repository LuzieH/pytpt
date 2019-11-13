import numpy as np
import matplotlib.pyplot as plt
import transition_paths as tp
import os.path


# load example data
my_path = os.path.abspath(os.path.dirname(__file__))
# row stochastic transition matrix
P = np.load(os.path.join(my_path, 'data/P_3well.npy'))
# index set of A,B,C
ind_A = np.load(os.path.join(my_path, 'data/ind_A_3well.npy'))
ind_B = np.load(os.path.join(my_path, 'data/ind_B_3well.npy'))
ind_C = np.load(os.path.join(my_path, 'data/ind_C_3well.npy'))
# dimension of the state space -> needed for plotting
S_shape = np.load(os.path.join(my_path, 'data/S_shape_3well.npy'))

# instanciate
# discretized triple well diffusion process
# the set A is the deep well on the LHS, the set B is the deep well on the RHS
triplewell = tp.transitions_mcs(P, ind_A, ind_B, ind_C)

# compute committor probabilities
[q_f, q_b] = triplewell.committor()

fig2 = plt.figure()
plt.imshow(np.reshape(q_f, (S_shape)), extent=[-2, 2, -2.5, 2.5])
plt.title('Forward committor')
plt.colorbar()

fig3 = plt.figure()
plt.imshow(np.reshape(q_b, (S_shape)), extent=[-2, 2, -2.5, 2.5])
plt.title('Backward committor')
plt.colorbar()

# therof compute the reactive density
reac_dens = triplewell.reac_density()

# and reactive currents
[current, eff_current] = triplewell.reac_current()
current_dens = triplewell.current_density()

fig4 = plt.figure()
plt.imshow(np.reshape(reac_dens, (S_shape)), extent=[-2, 2, -2.5, 2.5])
plt.title('Reactive density')
plt.colorbar()

fig5 = plt.figure()
plt.title('Reactive channels')
plt.imshow(np.reshape(current_dens, (S_shape)), extent=[-2, 2, -2.5, 2.5])
plt.colorbar()

# discrete rate = prob. that a transition trajectory leaves A during one time step
rate = triplewell.transition_rate()
print('The discrete rate is ' + str(rate))
