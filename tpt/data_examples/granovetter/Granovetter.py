import pickle
import os
import numpy as np
from networkx import number_of_nodes, from_numpy_matrix #, neighbors

class micromodel_stochastic():
    """Contains the microscopic Granovetter-like simulation model"""

    def __init__(self, A,  always_active_nodes, always_inactive_nodes, initially_active_nodes, 
                 initially_inactive_nodes, threshold=0.5, p_active=0.8, 
                 p_inactive=0.8, e_active=0.1, e_inactive=0.1,
                 load_cache=False, output_folder="calculations/"):
        """
        Initialize an instance of the Granovetter Micromodel.

        Parameters:
        -----------
        A: np. matrix
            adjacency matrix of the unweighted/undirected network 
            (no loops, connected network) 
        always_active_nodes : array of indices
            The indices of active nodes. This corresponds to the
            number of certainly active nodes during the whole simulation.
        always_inactive_nodes : array of indices
            The indices of inactive nodes. This corresponds to the
            number of certainly inactive nodes during the whole simulation.
        initially_active_nodes : array of indices
            The nodes that can change their state during the simulation and are 
            initially active.
        initially_inactive_nodes : array of indices
            The nodes that can change their state during the 
            simulation and are initially inactive.    
        threshold : float
            The individual threshold above which nodes become active with 
            probability p_active, below which nodes become inactive with 
            probability p_inactive. Must be between 0 and 1
        p_active: float between 0 and 1
            probability for an inactive node to become active when the threshold
            of active neighbours is reached 
        p_inactive: float between 0 and 1
            probability of becoming inactive when the threshold of inactive 
            neighbours is reached
        e_active: float between 0 and 1
            exploration probability for inactive nodes to become active
        e_inactive: float between 0 and 1
            exploration probability for active nodes to become inactive            
        load_cache : bool
            If true, load previously generated data for the specified model
            configuration from the disk and append newly generated data to this
            file.
        output_folder : str
            Relative or absolute path to the desired output folder for the
            calculations.
        """
 

        if output_folder[-1] != "/":
            output_folder += "/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._p_active = p_active
        self._p_inactive = p_inactive
        self._e_active = e_active
        self._e_inactive = e_inactive
        self._network  = A
        self._graph = from_numpy_matrix(A)
        self._number_of_nodes=number_of_nodes(self._graph)

        self._active_nodes = always_active_nodes
        self._inactive_nodes = always_inactive_nodes
        self._initially_active_nodes = initially_active_nodes
        self._initially_inactive_nodes = initially_inactive_nodes
        self._threshold = threshold

        filename = "granovetter_N{0}_NA{1}_NI{2}_T{3}_P{4}_E{5}.py".format(
              self._number_of_nodes,  np.size(self._active_nodes),  np.size(self._inactive_nodes), 
              threshold, p_active, e_active)

        self._filename = output_folder + filename
 

        if load_cache and os.path.exists(self._filename):
            _cache_dic = np.load(self._filename)
            self._results = _cache_dic["results"]
            self._run_id = _cache_dic["number_of_runs"]
        else:
            self._results = {}
            self._run_id = 0

 
 

    def run(self, max_iter, save_step=10, block_sizes=None):
        """Run the model once for the given network.

        Parameters:
        -----------        
        max_iter: int
            maximum number of iterations/time steps
        save_step: int
            saves only every save_steps'th step
        block_sizes: None or np.array
            sizes of the blocks, we will then store the number of active nodes 
            in time for each block individually
        """
        
        self._max_iter=max_iter
        
        print("Starting run:", self._run_id)


        if block_sizes is None:
            #store the number of active nodes of the whole graph
            block_sizes=np.array([self._number_of_nodes])

        # Turn lists into sets
        active_nodes_set = set(self._active_nodes) # add nodes that are always active
        inactive_nodes_set= set(self._inactive_nodes) #add nodes that are always inactive
        
        #add the initial conditions to the inactive/active set
        active_nodes_set.update(self._initially_active_nodes)
        inactive_nodes_set.update(self._initially_inactive_nodes)                   

        #nodes that can change their state, influenced by neighbours
        contingent_nodes_set=set()
        contingent_nodes_set.update(self._initially_active_nodes)
        contingent_nodes_set.update(self._initially_inactive_nodes)
  
        active_nodes_set_new=active_nodes_set.copy()
        inactive_nodes_set_new=inactive_nodes_set.copy() 
        

        # Store the time series of active nodes R(t) in each given subset of nodes
        blocks_currently_active = np.zeros((int(np.ceil(self._max_iter/save_step))+1,int(np.shape(block_sizes)[0])))
        fine_result = np.zeros(np.shape(block_sizes)[0])
        blocks_cumsum=np.zeros(np.shape(block_sizes)[0]+1)
        blocks_cumsum[1:]=np.cumsum(block_sizes)
        for j in range(np.shape(block_sizes)[0]):
            fine_result[j] = len(active_nodes_set.intersection(set(range(int(blocks_cumsum[j]), int(blocks_cumsum[j+1])))))       
        blocks_currently_active[0,:] = fine_result
        

        iter=1
        save_iter =1
        
        while True:
        
            for node in contingent_nodes_set:

                # Find the neighbors of the node
                nbs_set = set(self._graph.neighbors(node))

                # Find those neighbors that are active
                number_active = len([i for i in nbs_set if i in active_nodes_set])

                node_degree = self._graph.degree(node)
                active_share = float(number_active) / node_degree
                
                #if the active_share== threshold, let a coin decide 
                coin=np.random.rand()
                
                if active_share > self._threshold or (active_share == self._threshold and coin<0.5):
                    #if active -> with prob e_inactive explore
                    if node in active_nodes_set:
                        if np.random.rand()<self._e_inactive:
                            inactive_nodes_set_new.add(node)
                            active_nodes_set_new.remove(node)
                    #if inactive -> with prob p_active become active
                    else: 
                        if np.random.rand()<self._p_active:
                            active_nodes_set_new.add(node)
                            inactive_nodes_set_new.remove(node)
                            
                   
                else: #if active_share < self._threshold or == and  coin >=0.5
                    #if active -> become inactive with prob p_inactive
                    if node in active_nodes_set:
                        if np.random.rand()<self._p_inactive:
                            inactive_nodes_set_new.add(node)
                            active_nodes_set_new.remove(node)          
                    #if inactive -> become active with prob e_active
                    else:
                        if np.random.rand()<self._e_active:
                            active_nodes_set_new.add(node)
                            inactive_nodes_set_new.remove(node)                  
                

            active_nodes_set=active_nodes_set_new.copy()
            inactive_nodes_set=inactive_nodes_set_new.copy()  
            
            
            
            if np.mod(iter,save_step)==1:
 
                fine_result = list([0]*np.shape(block_sizes)[0])
                blocks_cumsum=np.zeros(np.shape(block_sizes)[0]+1)
                blocks_cumsum[1:]=np.cumsum(block_sizes)
                for j in range(np.shape(block_sizes)[0]):
                    fine_result[j] = len(active_nodes_set.intersection(set(range(int(blocks_cumsum[j]), int(blocks_cumsum[j+1])))))
             
                blocks_currently_active[save_iter,:] = fine_result
                    
                save_iter = save_iter+1
        

            # final condition
            if iter==self._max_iter-1:
                self._results[self._run_id] = blocks_currently_active
                self._run_id += 1
                return blocks_currently_active
                
 
           
            iter=iter+1

    def save(self):
        """ Save all current results to the disk."""
        results = {}
        results["number_of_nodes"] = self._number_of_nodes
        results["nbr_initially_active_nodes"] = np.size(self._initially_active_nodes)
        results["nbr_initially_inactive_nodes"] = np.size(self._initially_inactive_nodes)
        results["nbr_always_active_nodes"] = np.size(self._active_nodes)
        results["nbr_always_inactive_nodes"] = np.size(self._inactive_nodes)      
        results["number_of_runs"] = self._run_id
        results["threshold"] = self._threshold
        results["p_active"]= self._p_active 
        results["p_inactive"] = self._p_inactive 
        results["e_active"] =self._e_active 
        results["e_inactive"] =self._e_inactive         
        results["results"] = self._results
        pickle.dump(results, open(self._filename, "wb"))


    def current_number_of_runs(self):
        """The number of succesfully finished ensemble runs."""
        return self._run_id


 
