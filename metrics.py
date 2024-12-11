import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

class GraphEvaluation:
    @staticmethod
    def modularity(graph, communities):
        """
        Calculate the modularity of a graph given its communities.

        :param graph: NetworkX graph.
        :param communities: Dictionary where keys are community labels and values are lists of nodes.
        :return: Modularity score.
        """
        community_sets = {}
        for node, community in communities.items():
            if community not in community_sets:
                community_sets[community] = set()
            community_sets[community].add(node)

        # Convert to a list of sets
        community_list = list(community_sets.values())
        return nx.algorithms.community.modularity(graph, community_list)

    @staticmethod
    def accuracy(predicted_labels, true_labels):
        """
        Calculate the accuracy of predicted labels against true labels.

        :param predicted_labels: Dictionary of node -> predicted community label.
        :param true_labels: Dictionary of node -> true community label.
        :return: Accuracy score.
        """
        correct = sum(1 for node in true_labels if true_labels[node] == predicted_labels.get(node, -1))
        return correct / len(true_labels)

    @staticmethod
    def normalized_mutual_information(predicted_labels, true_labels):
        """
        Calculate the Normalized Mutual Information (NMI) between predicted and true labels.

        :param predicted_labels: Dictionary of node -> predicted community label.
        :param true_labels: Dictionary of node -> true community label.
        :return: NMI score.
        """
        true = [true_labels[node] for node in true_labels]
        predicted = [predicted_labels.get(node, -1) for node in true_labels]
        return normalized_mutual_info_score(true, predicted)

    @staticmethod
    def evaluate(graph, communities, true_labels):
        """
        Evaluate the graph coloring or community detection results using multiple metrics.

        :param graph: NetworkX graph.
        :param communities: Dictionary where keys are community labels and values are lists of nodes.
        :param true_labels: Dictionary of node -> true community label.
        :return: Dictionary of evaluation metrics.
        """
        # Flatten communities to create predicted labels
        predicted_labels = {node: community for community, nodes in communities.items() for node in nodes}

        return {
            "Modularity": GraphEvaluation.modularity(graph, communities),
            "Accuracy": GraphEvaluation.accuracy(predicted_labels, true_labels),
            "NMI": GraphEvaluation.normalized_mutual_information(predicted_labels, true_labels)
        }
