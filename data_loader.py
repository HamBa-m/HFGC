import pandas as pd
import networkx as nx

class GraphLoader:
    @staticmethod
    def load_from_csv(filepath):
        """
        Load a graph from a CSV file.

        :param filepath: Path to the CSV file with columns 'node_1' and 'node_2'.
        :return: A NetworkX graph.
        """
        df = pd.read_csv(filepath)
        G = nx.Graph()
        G.add_edges_from(df.values)
        return G