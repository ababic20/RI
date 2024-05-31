import matplotlib.pyplot as plt
import networkx as nx

# Kreiranje usmerenog grafa
G = nx.DiGraph()

# Dodavanje čvorova za svaki sloj
input_nodes = ['x1', 'x2']
hidden_nodes = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
output_nodes = ['y1', 'y2']

# Dodavanje svih čvorova u graf
G.add_nodes_from(input_nodes + hidden_nodes + output_nodes)

# Definisanje pozicija za svaki čvor
pos = {}
layer_x = [0, 1, 2]
layer_y_input = [0, 1]
layer_y_hidden = [i for i in range(len(hidden_nodes))]
layer_y_output = [0, 1]

pos.update({f'x{i+1}': (layer_x[0], layer_y_input[i]) for i in range(len(input_nodes))})
pos.update({f'h{i+1}': (layer_x[1], layer_y_hidden[i]) for i in range(len(hidden_nodes))})
pos.update({f'y{i+1}': (layer_x[2], layer_y_output[i]) for i in range(len(output_nodes))})

# Dodavanje ivica sa težinama
weights_input_hidden = {
    ('x1', 'h1'): 0.2, ('x1', 'h2'): 0.8, ('x1', 'h3'): 0.5, ('x1', 'h4'): 0.6, ('x1', 'h5'): 0.1, ('x1', 'h6'): 0.9,
    ('x2', 'h1'): 0.3, ('x2', 'h2'): 0.9, ('x2', 'h3'): 0.1, ('x2', 'h4'): 0.4, ('x2', 'h5'): 0.7, ('x2', 'h6'): 0.5
}

weights_hidden_output = {
    ('h1', 'y1'): 0.7, ('h1', 'y2'): 0.2, ('h2', 'y1'): 0.5, ('h2', 'y2'): 0.8, ('h3', 'y1'): 0.4, ('h3', 'y2'): 0.3,
    ('h4', 'y1'): 0.6, ('h4', 'y2'): 0.1, ('h5', 'y1'): 0.9, ('h5', 'y2'): 0.6, ('h6', 'y1'): 0.2, ('h6', 'y2'): 0.4
}

# Dodavanje ivica u graf
G.add_edges_from(weights_input_hidden.keys())
G.add_edges_from(weights_hidden_output.keys())

# Crtež grafa
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

# Crtež oznaka na ivicama
edge_labels_input_hidden = {(src, dst): f'{weight:.2f}' for (src, dst), weight in weights_input_hidden.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_input_hidden, font_color='red')

edge_labels_hidden_output = {(src, dst): f'{weight:.2f}' for (src, dst), weight in weights_hidden_output.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_hidden_output, font_color='green')

plt.title("Topologija neuronske mreže")
plt.show()
