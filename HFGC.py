import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import time
import tqdm

class HierarchicalFuzzyGraphColoring:
    """
    Class for the Hierarchical Fuzzy Graph Coloring algorithm.
    
    Parameters
    ----------
    k : int
    """
    def __init__(self, k):
        """
        Initialize the graph coloring algorithm.
        
        Parameters
        ----------
        k : int
        """
        self.k = k

    def _initialize_graph(self, graph):
        """
        Initialize the graph structure, nodes, and sorted nodes.
        
        Parameters
        ----------
        graph : networkx.Graph 
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.sorted_nodes = sorted(self.nodes, key=lambda n: graph.degree(n), reverse=True)
        # self.sorted_nodes = sorted(self.nodes, key=lambda n: nx.betweenness_centrality(graph)[n], reverse=True)

        
    def _initialize_masters(self, ground_truth):
        """
        Initialize master nodes based on the highest degree nodes by community if there is a ground truth.
        """
        if ground_truth is not None:
            self.masters = []
            for i in range(1, self.k + 1):
                nodes = [node for node in self.sorted_nodes if ground_truth[node] == i]
                self.masters.extend(nodes[:1])
        else:
            self.masters = self.sorted_nodes[:self.k]

    def _initialize_masters_and_slaves(self, ground_truth=None):
        """
        Initialize master and slave nodes.
        """
        self._initialize_masters(ground_truth)
        self.slaves = self.sorted_nodes[self.k:]

    def _initialize_labels(self):
        """
        Initialize labels for slave and master nodes.
        """
        self.labels = {node: np.zeros(self.k, dtype=int) for node in self.slaves}
        for i, node in enumerate(self.masters):
            self.labels[node] = np.eye(self.k)[i]
        self.master_colors = {master: i + 1 for i, master in enumerate(self.masters)}

    def _propagate_colors(self, next_colored_nodes, updated_labels, verbose):
        """
        Propagate colors to uncolored nodes and update labels.
        
        Parameters
        ----------
        next_colored_nodes : set
        updated_labels : dict
        verbose : bool
        """
        if verbose:
            for node in tqdm.tqdm(next_colored_nodes, desc="Coloration de nouveaux noeuds..."):
                neighbor_labels = [self.labels[neighbor] for neighbor in self.graph.neighbors(node)]
                total_label = np.sum(neighbor_labels, axis=0)
                updated_labels[node] = total_label
        else:
            for node in next_colored_nodes:
                neighbor_labels = [self.labels[neighbor] for neighbor in self.graph.neighbors(node)]
                total_label = np.sum(neighbor_labels, axis=0)
                updated_labels[node] = total_label

    def _propagate_between_slaves(self, colored_nodes, updated_labels, verbose):
        """
        Propagate colors between already colored slave nodes.
        
        Parameters
        ----------
        colored_nodes : set
        updated_labels : dict
        verbose : bool
        """
        stop = True
        if verbose:
            for node in tqdm.tqdm(colored_nodes, desc="Propagation de couleurs..."):
                if node in self.masters:
                    continue
                neighbor_labels = [self.labels[neighbor] for neighbor in self.graph.neighbors(node)]
                total_label = np.sum(neighbor_labels, axis=0)
                updated_labels[node] = total_label + self.labels[node]
        else:
            for node in colored_nodes:
                if node in self.masters:
                    continue
                neighbor_labels = [self.labels[neighbor] for neighbor in self.graph.neighbors(node)]
                total_label = np.sum(neighbor_labels, axis=0)
                updated_labels[node] = total_label

    def visualize_coloration(self):
        """
        Visualize the graph coloration process.
        """
        color_map = []
        for node in self.graph.nodes():
            if node in self.master_colors:
                color_map.append(self.master_colors[node])
            else:
                node_label = self.labels.get(node, np.zeros(self.k, dtype=int))
                color = np.argmax(node_label) + 1 if np.any(node_label) else 0
                color_map.append(color)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color=color_map, cmap=plt.cm.rainbow, node_size=500)
        plt.show()

    def run(self, graph, return_time=False, verbose=False):
        """
        Execute the graph coloring algorithm.
        """
        self._initialize_graph(graph)
        self._initialize_masters_and_slaves()
        self._initialize_labels()

        slaves_uncolored = len(self.slaves)
        just_colored_nodes = set(self.masters)
        colored_nodes = set(self.masters)

        t_start = time.time()
        
        while slaves_uncolored > 0 :
            updated_labels = copy.deepcopy(self.labels)
            next_colored_nodes = set()

            for node in just_colored_nodes:
                next_colored_nodes.update(self.graph.neighbors(node))

            next_colored_nodes.difference_update(colored_nodes)
            slaves_uncolored -= len(next_colored_nodes)

            self._propagate_colors(next_colored_nodes, updated_labels, verbose)
            self._propagate_between_slaves(colored_nodes, updated_labels, verbose)

            just_colored_nodes = next_colored_nodes
            colored_nodes.update(next_colored_nodes)
            self.labels = updated_labels

        t_end = time.time()

        return (self.labels, t_end - t_start) if return_time else self.labels
    
    def get_community_probabilities(self):
        """
        Get the community probabilities for each node.
        """
        probabilities = {}
        
        for node, label in self.labels.items(): 
            total = sum(label) 
            if total > 0:
                probs = [label[i] / total for i in range(self.k)] 
                probabilities[node] = probs 
            else:
                probabilities[node] = [1/self.k] * self.k 
        
        return probabilities
    
    def assign_communities(self):
        """
        Assign communities to each node.
        """
        probabilities = self.get_community_probabilities()
        communities = {}
        
        for node, node_probs in probabilities.items():
            community = np.argmax(node_probs) + 1
            communities[node] = community
        
        return communities