import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# calculate the minus cross entropy
def mins_cross_entropy(dist1, dist2):
    """
    Compute the cross-entropy between two distributions.

    Parameters:
    dist1 (torch.Tensor): The first distribution.
    dist2 (torch.Tensor): The second distribution.

    Returns:
    float: The cross-entropy value.
    """
    # Ensure the distributions are normalized and convert to log probabilities
    eps = 1e-7
    dist1 = torch.log(dist1+eps)
    
    # Calculate the cross-entropy
    cross_entropy_value = torch.mean(torch.sum(dist2 * dist1,dim=1))

    return cross_entropy_value


# shuffle rows, used for calculating the cross entropy for the grouping results
def shuffle_rows(matrix):
    """
    Shuffle the rows of a PyTorch matrix.

    Parameters:
    matrix (torch.Tensor): An N x d PyTorch matrix.

    Returns:
    torch.Tensor: The matrix with rows shuffled.
    """
    # Get the number of rows
    num_rows = matrix.size(0)

    # Create a random permutation of row indices
    shuffled_indices = torch.randperm(num_rows)

    # Shuffle the rows using the indices
    shuffled_matrix = matrix[shuffled_indices]

    return shuffled_matrix

# visualization of a graph
def visualize_weighted_graph(adj_matrix, node_sizes):
    """
    Visualize a weighted undirected graph.

    Parameters:
    adj_matrix (numpy.ndarray): A symmetric adjacency matrix representing the weighted edges.
    node_sizes (list): A list of sizes for each node.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Get weights for each edge
    weights = np.array([data['weight'] for _, _, data in G.edges(data=True)])*2

    # Draw the graph
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, width=weights, node_size=node_sizes, edge_color=weights, edge_cmap=plt.cm.Blues)
    plt.show()
