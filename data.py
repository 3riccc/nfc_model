import numpy as np
import networkx as nx


import torch
import torch_geometric


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def standardize_matrix(matrix):
    # Mean and standard deviation for each dimension
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    # Standardize the matrix
    eps=1e-7
    standardized_matrix = (matrix - mean) / (std+eps)

    return standardized_matrix

# get the feature matrix from a network
def get_feature(G):
    degrees = np.array(list(dict(G.degree()).values())).astype('float')[:,None]
    degree_diffs = []
    degree_chis = []
    
    for node in G.nodes():
        # Degree of the node
        node_degree = G.degree(node)

        # Degrees of neighbors
        neighbors_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]

        # Mean and standard deviation of neighbors' degrees
        mean_neighbors_degree = np.mean(neighbors_degrees) if neighbors_degrees else 0
        dc = (mean_neighbors_degree-node_degree)**2/mean_neighbors_degree
        degree_chis.append(dc)
        
        degree_diffs.append(node_degree-mean_neighbors_degree)
    degree_chis = np.array(degree_chis)[:,None]
    degree_diffs = np.array(degree_diffs)[:,None]
    clustering_coefficients = np.array(list(nx.clustering(G).values()))[:,None]
    core_numbers = np.array(list(nx.core_number(G).values())).astype('float')[:,None]

    node_features = np.concatenate((degrees,degree_diffs,degree_chis,clustering_coefficients,core_numbers),axis=1)
    
    normed_feas = standardize_matrix(node_features)
    return torch.from_numpy(normed_feas)

#  get the partition function
def get_zt(padj):
    padj = padj*(1-torch.eye(padj.shape[0]).to(device))
    ts = [0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24]
    cal_points = len(ts)
    ts = torch.tensor(ts).unsqueeze(1).repeat(1,padj.shape[0]).to(device)
    
    D = torch.sum(padj,dim=1).unsqueeze(1).repeat(1,padj.shape[0])
    D = D*torch.eye(D.shape[0]).to(device)
    L = D-padj
    evl,evc = torch.linalg.eig(L)
    evl = evl.unsqueeze(0).repeat(cal_points,1)
    
    zts = torch.exp(-evl*ts)
    zts = torch.sum(zts,dim=1) / padj.shape[0]
    return zts