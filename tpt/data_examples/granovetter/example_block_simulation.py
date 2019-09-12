"""
Created on Mon Aug 12 10:18:52 2019

@author: Luzie Helfmann, helfmann@zib.de
"""
from networkx import from_numpy_matrix#, draw
import numpy as np
import Granovetter 
import stochastic_block_model
import matplotlib.pyplot as plt
import os.path

#%matplotlib inline
#%matplotlib qt


#########################################
#network


load_data=True


my_path = os.path.abspath(os.path.dirname(__file__))
#cluster sizes
sizes = np.array([50,50,50,50,50])

if load_data:
    #load
     
   
    A = np.load(os.path.join(my_path, 'networks/A2.npy'))
    G = from_numpy_matrix(A)

else: 
    #Create block model graph

    #symmetric wiring probabiliies between clusters
    probs = np.array([[0.1,0.02,0,0.01,0],
             [0.02,0.1,0.02,0,0],
             [0,0.02,0.1,0,0.02],
             [0.01,0,0,0.1,0.01],
             [0,0,0.02,0.01,0.1]])
    
    #create block model graph
    A = stochastic_block_model.adjacency(sizes, probs)
    #create networkx graph
    G = from_numpy_matrix(A)
    #save graph
    np.save(os.path.join(my_path, 'networks/A4'),A )
    
#plt.figure()
#draw(G)


########################################
#Granovetter stochastic micro model

#initial conditions
#assign the nodes that are always active of inactive
active_nodes=np.arange(40) 
inactive_nodes=np.arange(243,250)
#the nodes that are only initially active are drawn randomly from the remaining nodes
jrandom = np.random.choice(np.arange(0,203),1)[0] #random number of initially active nodes
initially_active_nodes= np.random.choice(np.arange(40,243),jrandom,replace=False)
initially_inactive_nodes=np.array([i for i in np.arange(40,243) if i not in initially_active_nodes])

#micromodel instantiation
block=Granovetter.micromodel_stochastic(A,  active_nodes, inactive_nodes, initially_active_nodes, initially_inactive_nodes,e_active=0.1,e_inactive=0.1,p_active=0.9,p_inactive=0.9)

#number of time steps
maxtime=2000000
save_step=1000
#run simulation
R=block.run(maxtime,save_step=save_step,block_sizes=sizes)

#########################################
#Plotting

#plotting the number of active nodes in each block vs time
fig,ax = plt.subplots(nrows=6, ncols=1, figsize=(20,10), sharex=True, squeeze=True)
ax[0].set_title('Number of active nodes')

#in each block
R_blocks=np.array(R)
for i in range(5):
    ax[i].plot(R_blocks[:,i],label='R_'+str(i+1)+'(t)') 
    ax[i].legend()

#total number of active nodes
R_all=np.sum(R_blocks,axis=1)
ax[5].plot(R_all,label='R(t)')
ax[5].legend()
ticks= np.arange(0,np.shape(R)[0],2000)
ax[5].set_xticks(ticks) 
ax[5].set_xticklabels(ticks*save_step)
ax[5].set_xlabel('t')


#histogram of the macro states (how many active nodes)
plt.figure()
plt.hist(R_all,250, range=(0,250))
plt.title('Histogram of R(t)')

plt.figure()
plt.scatter(R_all[:-1],R_all[1:],s=0.1)
plt.xlim(0,250)
plt.ylim(0,250)
plt.title('R(t) vs. R(t+1)')
plt.xlabel('R(t)')
plt.ylabel('R(t+1)')
