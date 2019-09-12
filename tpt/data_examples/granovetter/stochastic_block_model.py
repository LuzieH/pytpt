"""
Created on Fri Aug  9 14:27:03 2019

@author:  Luzie Helfmann, helfmann@zib.de
"""

import numpy as np


def adjacency(clustersizes, wiring_probs, connected=True):
    '''
    Function that returns the adjacency matrix of a stochastic block model network, 
    if needed it can be enforced that the network is connected. 
    
    Parameters:
    -----------------
    clustersizes: np.array
        array of the number of nodes in each cluster, if there are |C| clusters, 
        then the array is of size |C|
    wiring_probs: np.array
        matrix of size |C|x|C| that contains the probabilities of edges between 
        a node in cluster C_i and a node in cluster C_j
    connected: boolean
        if True, a connected network is enforced (actually, it is only check that
        there are no single unconnected nodes)
        
    '''
    
    #number of nodes
    N =  np.sum(clustersizes)
    #number of clusters |C|
    N_clusters=np.size(clustersizes)
    
    cluster_member=np.zeros(N) #vector of the assigned cluster for each node
    for j in range(N_clusters):
        cluster_member[np.sum(clustersizes[:j]):np.sum(clustersizes[:j+1])]=j
        
    #adjacency matrix    
    A=np.zeros((N,N))
    for i in range(N):
        for j in np.arange(i+1,N): #no self-loops
            
            #wire with the corresponding probability
            if np.random.rand()<wiring_probs[int(cluster_member[i]),int(cluster_member[j])]:
                A[i,j]=1
                A[j,i]=1 #since the network is undirected -> symmetric matrix
    
    #if needed, enforce connectivity   
    #actually, we only check if there are single unconnected nodes!!    
    single_nodes=np.where(np.sum(A,axis=1)==0)[0]
    s_size = np.size(single_nodes)   
    
    if connected==True and s_size>0:      
        while s_size>0:
            #choose the cluster with which to rewire
            probs=wiring_probs[int(cluster_member[single_nodes[0]]),:]/np.sum(wiring_probs[int(cluster_member[single_nodes[0]]),:])
            chosen_cluster = np.random.choice(np.arange(N_clusters),1,p=probs)[0]
            #choose a random node from that cluster
            chosen_node=np.random.choice(clustersizes[chosen_cluster],1)[0] + np.cumsum(clustersizes)[chosen_cluster] -clustersizes[chosen_cluster]
            
            A[single_nodes[0],chosen_node]=1
            A[chosen_node,single_nodes[0]]=1
            
            single_nodes=np.where(np.sum(A,axis=1)==0)[0]
            s_size = np.size(single_nodes)              
        
        
    return A
    