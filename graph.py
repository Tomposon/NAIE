from utils import read_feature, read_graph, read_label
import networkx as nx
import numpy as np
import scipy.sparse as sp
import os

class Graph(object):
    def __init__(self, features, G, labels=None):
        self.features = features
        self.G = G
        self.labels = labels
    
    def adjcency_matrix(self, is_sparse=False):
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=sorted(self.G.nodes()))

        if is_sparse:
            return adj_matrix
        
        return adj_matrix.toarray().astype(np.float64)
    
    def nodes(self):
        return self.G.nodes()

    def node_num(self):
        return self.G.number_of_nodes()

    def normalized_adjmatrix(self):
        adj_matrix = nx.adjacency_matrix(self.G, nodelist=sorted(self.G.nodes())).toarray().astype(np.float64)
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
        diag = np.sum(adj_matrix, axis=1)
        diag = np.power(diag, -0.5)
        degree_matrix = np.diag(diag)
        degree_matrix = np.matrix(degree_matrix)
        adj_matrix = np.matrix(adj_matrix)
        
        return degree_matrix * adj_matrix * degree_matrix

def load_graph(dataset, labels_is_onehot=True):
    features = read_feature("./data/" + dataset + ".feature", is_normalize=False)

    if os.path.exists("./data/" + dataset + ".label"):
        labels = read_label("./data/" + dataset + ".label", is_onehot=labels_is_onehot)
    else:
        labels = None

    G = read_graph("./data/" + dataset + '.edgelist')

    graph = Graph(features, G, labels)

    return graph
  