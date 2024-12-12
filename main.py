from HFGC import HierarchicalFuzzyGraphColoring as HFGC
from data_loader import GraphLoader
from metrics import GraphEvaluation
import os
import json
from tqdm import tqdm

results = {}

# Get all CSV files in the data/ folder
csv_files = [f for f in os.listdir('data/') if f.endswith('.csv')]

# Define weight settings
weight_settings = [
    None,  # No weights
    [1, 0, 0, 0], # degree centrality
    [0, 1, 0, 0], # closeness centrality
    [0, 0, 1, 0], # betweenness centrality
    [0, 0, 0, 1], # eigenvector centrality
    [0.1, 0.4, 0.3, 0.2]  # random distribution
]

# Iterate over each CSV file
for csv_file in tqdm(csv_files, desc="CSV Files"):
    if csv_file == 'new_sites_edges.csv':
        continue
    file_results = {}
    G = GraphLoader.load_from_csv(f'data/{csv_file}')
    print(f"Loaded graph with {len(G)} nodes and {len(G.edges)} edges.")
    hfgc = HFGC(k=20)
    hfgc.initialize_graph(G)
    
    for weights in tqdm(weight_settings, desc="Weight Settings", leave=False):
        hfgc.set_weights(weights)
        (_, t) = hfgc.run(return_time=True)
        communities = hfgc.assign_communities()
        mod = GraphEvaluation.modularity(G, communities)
        file_results[str(weights)] = {'modularity': mod, 'time': t}
    
    # Find the best weight setting
    best_weights = max(file_results, key=lambda w: file_results[w]['modularity'])
    best_modularity = file_results[best_weights]['modularity']
    
    # Test different values of k with the best weight setting
    k_results = {}
    t_results = {}
    for k in tqdm(range(2, 51), desc="k Values", leave=False):
        hfgc.set_k(k)
        hfgc.set_weights(eval(best_weights))
        (_ , t) = hfgc.run(return_time=True)
        communities = hfgc.assign_communities()
        mod = GraphEvaluation.modularity(G, communities)
        k_results[k] = mod
        t_results[k] = t
    
    results[csv_file] = {
        'all_weights_results': file_results,
        'best_weights': best_weights,
        'best_modularity': best_modularity,
        'k_results': k_results,
        't_results': t_results
    }

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)