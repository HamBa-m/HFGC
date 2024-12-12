import os
import time
import json
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.community import label_propagation_communities 
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import asyn_fluidc
from data_loader import GraphLoader

def process_datasets(input_folder="data", output_file="results_baselines.json", k=20):
    results = {}

    for filename in tqdm(os.listdir(input_folder), desc="Processing datasets"):
        if filename.endswith(".csv"):
            # Load the dataset as a graph
            G = GraphLoader.load_from_csv(f'data/{filename}')

            # Initialize result container for the dataset
            results[filename] = {}

            # Label Propagation Algorithm
            start_time = time.time()
            communities = list(label_propagation_communities(G))
            end_time = time.time()

            modularity_score = modularity(G, communities)
            results[filename]["label_propagation"] = {
                "modularity": modularity_score,
                "time": end_time - start_time,
            }

            # Fluid Communities Algorithm (if applicable)
            if k <= len(G.nodes):
                start_time = time.time()
                communities = list(asyn_fluidc(G, k))
                end_time = time.time()

                modularity_score = modularity(G, communities)
                results[filename]["fluid_communities"] = {
                    "modularity": modularity_score,
                    "time": end_time - start_time,
                }
            else:
                results[filename]["fluid_communities"] = "Not Applicable (k > number of nodes)"

    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    process_datasets()