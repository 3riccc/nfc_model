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
#     padj = padj*(1-torch.eye(padj.shape[0]).to(device))
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

#  get the partition function
def get_zt_ts(padj,ts):
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


def read_mtx(file_path):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for idx in range(len(lines)):
        if '%' not in lines[idx]:
            x = idx
            y = idx+1
            break
        
    
    # 第x行包含节点数，这里假设矩阵是方阵
    num_nodes = int(lines[x].split()[0])

    # 初始化行、列和数据列表
    rows, cols, data = [], [], []

    # 遍历文件的每一行，跳过第一行
    for line in lines[y:]:
        row, col = map(int, line.split()[:2])
        # 因为.mtx格式是从1开始计数，Python是从0开始，所以要减1
        rows.append(row - 1)
        cols.append(col - 1)
        data.append(1)  # 假设连接的权重为1

    nn = max(np.max(rows),np.max(cols))+1
    adj = np.zeros([nn,nn])
    adj[rows,cols] = 1
    G = nx.from_numpy_array(adj)
    
    # 创建稀疏矩阵的邻接矩阵
#     adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
#     # 将邻接矩阵转换为CSR格式以便高效查询
#     adjacency_matrix_csr = csr_matrix(adjacency_matrix)
#     G = nx.from_scipy_sparse_matrix(adjacency_matrix_csr)
    
    return G

def read_edges_file(file_path):
    with open(file_path, 'r') as file:
        edges = []
        for line in file:
            if '%' not in line:
                if ',' in line:
                    edges.append(line.strip().split(','))
                elif ' ' in line:
                    edges.append(line.strip().split())
#         edges = [ for line in file]

    # 假设节点从0开始编号
    max_node = max([max(int(edge[0]), int(edge[1])) for edge in edges])
    
    # 创建一个空的邻接矩阵
    adjacency_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)

    # 填充邻接矩阵
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # 如果是无向图，需要这一行
#     matrix = csr_matrix(adjacency_matrix)
    return nx.from_numpy_array(adjacency_matrix)