import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes for each layer
input_nodes = ['x1', 'x2', 'x3']
hidden_nodes = ['h1', 'h2', 'h3', 'h4']
output_nodes = ['y1', 'y2']

# Add all nodes to the graph
G.add_nodes_from(input_nodes + hidden_nodes + output_nodes)

# Define positions for each node
pos = {}
layer_x = [0, 1, 2]
layer_y_input = [0, 1, 2]
layer_y_hidden = [0, 1, 2, 3]
layer_y_output = [0, 1]

pos.update({f'x{i+1}': (layer_x[0], layer_y_input[i]) for i in range(len(input_nodes))})
pos.update({f'h{i+1}': (layer_x[1], layer_y_hidden[i]) for i in range(len(hidden_nodes))})
pos.update({f'y{i+1}': (layer_x[2], layer_y_output[i]) for i in range(len(output_nodes))})

# Add edges with weights (sample weights)
weights = {
    ('x1', 'h1'): 0.2, ('x1', 'h2'): 0.8, ('x1', 'h3'): 0.5, ('x1', 'h4'): 0.6,
    ('x2', 'h1'): 0.3, ('x2', 'h2'): 0.9, ('x2', 'h3'): 0.1, ('x2', 'h4'): 0.4,
    ('x3', 'h1'): 0.7, ('x3', 'h2'): 0.2, ('x3', 'h3'): 0.8, ('x3', 'h4'): 0.3,
    ('h1', 'y1'): 0.6, ('h1', 'y2'): 0.1, 
    ('h2', 'y1'): 0.7, ('h2', 'y2'): 0.9, 
    ('h3', 'y1'): 0.5, ('h3', 'y2'): 0.2, 
    ('h4', 'y1'): 0.8, ('h4', 'y2'): 0.4,
}

# Add edges to the graph
for (src, dst), weight in weights.items():
    G.add_edge(src, dst, weight=weight)

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

# Draw edge labels
edge_labels = {(src, dst): f'{weight:.2f}' for (src, dst), weight in weights.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Neural Network Topology")
plt.show()
