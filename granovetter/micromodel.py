import pickle
import os
import random
import numpy as np
from igraph import Graph


class Micromodel():
    """Contains the microscopic Granovetter-like simulation model"""

    def __init__(self, number_of_nodes, average_degree, network_type,
                 initially_active_nodes, threshold, potentially_active=None,
                 load_cache=False, rewiring_probability=None,
                 output_folder="calculations/"):
        """
        Initialize an instance of the Granovetter Micromodel.

        Parameters:
        -----------
        number_of_nodes : int
            The number of nodes in the network.
        average_degree : int
            The average degree of nodes in the network (must be even!)
        network_type : str
            The type of network that should be studied. Currently three options
            are implemented: ER (Erdos-Renyi), WS (Watts-Strogatz), BA
            (Barabasi-Albert)
        initially_active_nodes : int
            The number of initially active nodes. This corresponds to the
            number of certainly active nodes.
        threshold : float
            The individual threshold above which nodes become active. Must be
            between 0 and 1
        potentially_active : int
            The number of potentially active nodes. Must be larger than
            initially_active_nodes and smaller than number_of_nodes. If it is
            not set by the user, it is automatically set to its largest
            possible value.
        load_cache : bool
            If true, load previously generated data for the specified model
            configuration from the disk and append newly generated data to this
            file.
        rewiring_probability : float
            The Watts-Strogatz rewiring probability. Is only require if
            network_type="WS".
        output_folder : str
            Relative or absolute path to the desired output folder for the
            calculations.
        """
        assert (average_degree % 2 == 0), "Average degree must be even."

        if output_folder[-1] != "/":
            output_folder += "/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._average_degree = average_degree
        self._number_of_nodes = number_of_nodes
        self._network_type = network_type
        self._rewiring_probability = rewiring_probability
        self._initially_active_nodes = initially_active_nodes
        self._threshold = threshold
        self._graph = None

        filename = "granovetter_{0}_N{1}_K{2}_b{3}_A{4}_T{5}_P{6}.p".format(
            network_type, number_of_nodes, average_degree,
            rewiring_probability, initially_active_nodes, threshold,
            potentially_active)

        self._filename = output_folder + filename

        if not potentially_active:
            self._potentially_active_nodes = number_of_nodes
        else:
            self._potentially_active_nodes = potentially_active

        if load_cache and os.path.exists(self._filename):
            _cache_dic = np.load(self._filename)
            self._results = _cache_dic["results"]
            self._run_id = _cache_dic["number_of_runs"]
        else:
            self._results = {}
            self._run_id = 0


    def _erdos_renyi(self):
        """
        Initialize a new Erdos-Renyi network topology.

        The graph is not returned but stored as class attribute in self._graph.
        """
        print("Initializing Erdos-Renyi random network")
        number_of_links = self._number_of_nodes * self._average_degree // 2
        network = Graph.Erdos_Renyi(self._number_of_nodes, m=number_of_links)
        self._graph = network


    def _watts_strogatz(self):
        """
        Initialize a new Watts-Strogatz network topology.

        The graph is not returned but stored as class attribute in self._graph.
        """
        print("Initializing Watts-Strogatz random network")

        assert self._rewiring_probability, "Rewiring probability must be \
                given on init"
        nei = self._average_degree // 2
        network = Graph.Watts_Strogatz(dim=1, size=self._number_of_nodes,
                                       nei=nei,
                                       p=self._rewiring_probability, loops=False,
                                       multiple=False)
        self._graph = network


    def _barabasi_albert(self):
        print("Initializing Barabasi-Albert random network")
        number_of_links = self._average_degree // 2
        network = Graph.Barabasi(n=self._number_of_nodes, m=number_of_links)
        self._graph = network


    def _randomize(self):
        if self._network_type == "ER":
            self._erdos_renyi()
        elif self._network_type == "WS":
            self._watts_strogatz()
        elif self._network_type == "BA":
            self._barabasi_albert()


    def run(self):
        """Run the model once for a randomized initial setup."""
        print("Starting run:", self._run_id)

        # Set the seed as the current ID of the ensemble run
        random.seed(self._run_id)
        np.random.seed(self._run_id)

        # Create a new random network topology
        self._randomize()

        # Assign P potentially active nodes
        potentially_active_nodes = np.random.choice(
            range(self._number_of_nodes),
            size=self._potentially_active_nodes,
            replace=False)

        # Choose A certainly active out of the P potentially active nodes
        active_nodes = np.random.choice(potentially_active_nodes,
                                        size=self._initially_active_nodes,
                                        replace=False)

        # Turn both lists into sets for easier updating
        active_nodes = set(active_nodes)
        potentially_active_nodes = set(potentially_active_nodes)

        # Determine the set of contingent nodes of size C=P-A
        contingent_nodes = potentially_active_nodes - active_nodes

        # Store the time series of active nodes R(t)
        currently_active = [len(active_nodes)]

        # Finished nodes are those nodes that are active and only have active
        # neighbors, we store them for better performance
        finished_nodes = set()

        while True:
            # Start with searching for all contigent nodes that have at least
            # one active neighbor. That way we later only need to loop over all
            # contigent nodes that are close to the front of the cascade.
            # This procedure increases performance for large networks.
            contingent_with_active_nbs = set()

            # Search active nodes that have at least one contingent neighbor,
            # i.e., are not among the finished nodes
            for act in active_nodes.difference(finished_nodes):

                # Find the neighbors of an unfinished active node
                nbs = set(self._graph.neighbors(act))

                # Find those neighbors that are contigent
                nbs.intersection_update(contingent_nodes)

                # If there are no such neighbors, add the active node to the
                # set of finished nodes, so we don't check it again
                if not nbs:
                    finished_nodes.add(act)

                # Add the contingent neighbors to the set of nodes that may
                # possibly change its state in the current time step
                contingent_with_active_nbs.update(nbs)

            # The next two lines were here before I cleaned up the script, but
            # it seems not necessary, keep it here as a note for later
            # reference. I inserted the assert statement to make sure that
            # contingent_with_active_nbs.intersection_update(contingent_nodes)
            # really would have no effect
            #print(contingent_with_active_nbs.intersection_update(contingent_nodes))
            #contingent_with_active_nbs.intersection_update(contingent_nodes)
            assert contingent_with_active_nbs == \
                    contingent_with_active_nbs.intersection(contingent_nodes)

            # Now, determine which nodes become active in the current step
            newly_active = set()

            # Iterate over all contingent nodes that have at least one active
            # neighbor
            for contingent in contingent_with_active_nbs:
                nbs = self._graph.neighbors(contingent)
                number_active = len([i for i in nbs if i in active_nodes])

                node_degree = self._graph.degree(contingent)
                active_share = number_active / node_degree

                if active_share > self._threshold:
                    newly_active.add(contingent)

            # If noone becomes active in the last round or no more inactive
            # nodes remain we assume the cascade the cascade has stopped.  So
            # we return the time series of active nodes if full convergence is
            # not desired
            if (not newly_active) or (not contingent_nodes):
                self._results[self._run_id] = currently_active
                self._run_id += 1
                print("Result:", currently_active)
                return

            # Add the new nodes to the set of active nodes and remove them from
            # the set of inactive nodes
            active_nodes.update(newly_active)
            contingent_nodes.difference_update(newly_active)
            currently_active.append(len(active_nodes))


    def save(self):
        """ Save all current results to the disk."""
        results = {}
        results["number_of_nodes"] = self._number_of_nodes
        results["average_degree"] = self._average_degree
        results["rewiring_probability"] = self._rewiring_probability
        results["initially_active_nodes"] = self._initially_active_nodes
        results["number_of_runs"] = self._run_id
        results["threshold"] = self._threshold
        results["network_type"] = self._network_type
        results["results"] = self._results
        results["potentially_active"] = self._potentially_active_nodes
        pickle.dump(results, open(self._filename, "wb"))


    def current_number_of_runs(self):
        """The number of succesfully finished ensemble runs."""
        return self._run_id


if __name__ == "__main__":

    MODEL = Micromodel(number_of_nodes=10000, average_degree=10,
                       network_type="ER", initially_active_nodes=1000,
                       potentially_active=5000, threshold=0.2,
                       load_cache=False)

    for _ in range(5):
        MODEL.run()
        MODEL.save()
