import json
import networkx as nx
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt


# Load graph from JSON
def load_graph(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'], x=node['x'], y=node['y'])

    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])

    return G


# Convert to PyG Data format
def preprocess_graph(G):
    pyg_data = from_networkx(G)
    return pyg_data


# Plot graph for visualization
def plot_graph(G):
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=10)
    plt.show()


# Example usage
G = load_graph('ex1_k6_xr32.json')  # Use the uploaded file path
plot_graph(G)  # Visualize original graph
pyg_data = preprocess_graph(G)
