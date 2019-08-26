#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:57:51 2019

@author: bzfhelfm
"""
import numpy as np
import matplotlib.pyplot as plt
import transition_paths as tp

#set up a transition matrix for a random walker in a maze

S=40 

create_new_example=True 

if create_new_example ==True:
    #EITHER CREATE NEW EXAMPLE 
    pot=np.ones((40,40))
    pot[0,0]=0
    x=np.array([0,0])
    
    while x[0]!=39 or x[1]!=39:
        #horizontal or vertical movement? 
        w=np.random.choice(4,1)[0]
        #0 is vertical up, 1 is vertical down, 2 is horizontal left, 3 is horizontal right
        # when the move is not possible, stay where you were
        moves = np.array([np.array([-1,0]),np.array([1,0]),np.array([0,-1]),np.array([0,1])])
        if (x+moves[w])[0]>-1and (x+moves[w])[1]>-1 and (x+moves[w])[0]<40 and (x+moves[w])[1]<40:
            x=x+moves[w]
        pot[x[0],x[1]]=0
   
else: 
    #OR LOAD DATA
    T=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/Example_Maze/P_maze.npy') #transition matrix, row sum =1
    pot=np.load('/nfs/numerik/bzfhelfm/Dokumente/PhD/code/transitions/TPT_MCs/Example_Maze/pot_maze.npy')  
     

plt.imshow(pot)
plt.colorbar()

st=int(S*S-np.sum(pot))
xy_pot=np.zeros((st,2)) #all possible states -> gives the state space of size st
count=0
for i in range(S):
    for j in range(S):
        if pot[i,j]==0:
                xy_pot[count,:]=np.array([i,j])
                count=count+1
T=np.zeros((st,st))
for i in range(st):
    for j in np.arange(st):
        #unnormalized transitions
        if np.sum(np.abs(xy_pot[i,:]-xy_pot[j,:]))<2:
            T[i,j] = 1
            T[j,i] = 1
          
T= T/np.sum(T,axis=1).reshape(st,1)     

def solveStationary( A ):
    """ x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    return np.array(np.linalg.lstsq( a, b )[0])
       
stat_dens=solveStationary(T).flatten()

ind_A=np.array([0])
ind_B=np.array([st-1])
ind_C=np.arange(1,st-1)


maze=tp.transitions_mcs(T, ind_A, ind_B, ind_C, stat_dens = stat_dens)            
[q_f,q_b]=maze.committor()
#therof compute the reactive density
reac_dens=maze.reac_density()
#and reactive currents
[current,eff_current]=maze.reac_current()
current_dens=maze.current_density()

stat_dens_full = np.zeros((S,S))
q_f_full = np.zeros((S,S))
q_b_full  = np.zeros((S,S))
reac_dens_full  = np.zeros((S,S))
current_dens_full  = np.zeros((S,S))

count=0
for i in range(S):
    for j in range(S):
        if pot[i,j]==0:
            stat_dens_full[i,j]=stat_dens[count]
            q_f_full[i,j]=q_f[count]
            q_b_full[i,j]=q_b[count]
            reac_dens_full[i,j]=reac_dens[count]
            current_dens_full[i,j]=current_dens[count]
            count=count+1






fig1 = plt.figure()
plt.imshow(stat_dens_full)
plt.title('Stationary density')
plt.colorbar()

fig2 = plt.figure()
plt.imshow(q_f_full)
plt.title('Forward committor')
plt.colorbar()

fig3 = plt.figure()
plt.imshow(q_b_full)
plt.title('Backward committor')
plt.colorbar()

fig4 = plt.figure()
plt.imshow(reac_dens_full)
plt.title('Reactive density')
plt.colorbar()

 
fig5  = plt.figure()
plt.title('Reactive channels')
plt.imshow(current_dens_full)
plt.colorbar()