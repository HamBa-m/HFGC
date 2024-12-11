from HFGC import HierarchicalFuzzyGraphColoring
from data_loader import GraphLoader
from metrics import GraphEvaluation

# Initialize the Hierarchical Fuzzy Graph Coloring algorithm
hfgc = HierarchicalFuzzyGraphColoring(k=20)

# Load a graph from a CSV file
G = GraphLoader.load_from_csv('data/government_edges.csv')

# Run the Hierarchical Fuzzy Graph Coloring algorithm
hfgc.run(G, verbose=True)

# Convert the fuzzy communities to crisp communities
communities = hfgc.assign_communities()

# Evaluate the algorithm using modularity
mod = GraphEvaluation.modularity(G, communities)

# Print the modularity score
print(f'Modularity: {mod}')