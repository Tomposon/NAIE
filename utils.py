import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
import networkx as nx
import random

"""
data read
attribute network dataset often contain 3 parts：network topology, node attribute, node label
topology file extension：xx.edgelist
attribute file extension：xx.feature
label file extension：xx.label
"""

def read_feature(filename, sep=' ', is_normalize=False):

    f = open(filename)
    line = f.readline()

    node_num, feature_num = [int(_) for _ in line.strip('\n\r').split(sep)]

    features = []
    for line in f:
        feature = [float(value) for value in line.strip('\n\r').split(sep)]
        features.append(feature)

    features = np.array(features)

    if is_normalize == True:
        features[features > 0] = 1.0
    
    return features

def read_graph(filename):
    G = nx.read_edgelist(filename, create_using=nx.Graph(), nodetype=int)
    return G

def read_label(filename, sep=' ', is_onehot=True):
    f = open(filename)

    lines = f.readlines()
    node_num = len(lines)
    labels = np.zeros([node_num])
    label_set = set()
    for line in lines:
        node, label = [int(_) for _ in line.strip('\n\r').split(sep)]
        labels[node] = label
        label_set.add(label)
    label_num = len(label_set)
    label_map = dict(zip(label_set, range(label_num)))
    if is_onehot == True:
        onehot_labels = np.zeros([node_num, label_num])
        for node, label in enumerate(labels):
            onehot_labels[node, label_map[label]] = 1.0
        
        return onehot_labels

    return labels

def multiclass_classification(X, y, rnd, test_size=0.7):
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=rnd, test_size=test_size)
    model = LinearSVC()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    macro_f1 = f1_score(test_y, pred_y, average='macro')
    micro_f1 = f1_score(test_y, pred_y, average='micro')

    return macro_f1, micro_f1

def node_classification_evaluation(X, y, rnd=2019, test_size=0.7):
    macro, micro = 0., 0.
    times = 10
    mas = []
    mis = []
    for i in range(times):
        rnd_seed = np.random.randint(rnd)
        macro_f1, micro_f1 = multiclass_classification(X, y, rnd_seed, test_size)
        mas.append(macro_f1)
        mis.append(micro_f1)
        macro += macro_f1
        micro += micro_f1
    mas = np.array(mas)
    mis = np.array(mis)
    print(mas)
    print(np.var(mas))
    return macro / times, micro / times, np.std(mas), np.std(mis)

def cos_sim(a, b):
    return a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b)))

def link_prediction_evaluation(link_pairs, emb):
    cos_scores = []
    truth = []
    for data in link_pairs:
        cos_scores.append(cos_sim(emb[data[0]], emb[data[1]]))
        truth.append(data[2])
    
    auc = roc_auc_score(truth, cos_scores)
    if auc < 0.5:
        auc = 1- auc
    
    return auc


# produce gamma by small-worldness strategy 
def get_omega(nx_G):
    is_connected = nx.is_connected(nx_G)
    print("is connected ：{}".format(is_connected)) 
    components = list(nx.connected_components(nx_G))
    print("count of cc：{}".format(len(components)))
    
    if len(components) == 1:
        G = nx_G
    else:
        G = nx.Graph()
        node2id = dict()
        for id, node in enumerate(components[0]):
            node2id[node] = id
        
        for node1 in components[0]:
            for node2 in nx_G.adj[node1]:
                G.add_edge(node2id[node1], node2id[node2])
    
    avg_cluster = nx.average_clustering(G)
    print("clustering coefficient：{:.5f}".format(avg_cluster))
    short_path_length = nx.average_shortest_path_length(G)
    print("average shortest path length：{:.5f}".format(short_path_length))

    print("computing omega...")
    omega = nx.omega(G, niter=1, nrand=1)
    print("omega={:.5f}".format(omega))
    return omega
    

# produce gamma by node convergency strategy
def get_balance_coefficient(nx_G, X):
    cos_sims = cosine_similarity(X, X)
    print("Similarity Computing Finished!")
    rate = 0

    isolate_count = 0

    for node in nx_G.nodes():
        neis = list(nx.neighbors(nx_G, node))
        neis_of_neis = []
        for nei in neis:
            neis_of_neis.extend(list(nx.neighbors(nx_G, nei)))
        neis.extend(neis_of_neis)
    

        neis = list(set(neis))

        neis_sims = cos_sims[node, neis]
        if len(neis_sims) == 0:
            isolate_count = isolate_count + 1
            continue
        avg_of_nei_sims = np.min(neis_sims)
        #avg_of_nei_sims = neis_sims.median()
       
        non_neis = set(nx.non_neighbors(nx_G, node))
        non_neis = non_neis - set(neis_of_neis)
       
         
        non_neis = list(non_neis)
        non_neis_sims = cos_sims[node, non_neis]
        abnormals = np.where(non_neis_sims > avg_of_nei_sims)[0]
        
        
        num_of_abnormals = abnormals.shape[0]
        num_of_neis = len(neis)
        
        rate += num_of_abnormals / len(non_neis)
    return rate / (len(nx_G.nodes())-isolate_count)
    
def read_test_links(filename):
	f = open(filename)
	test_pairs = []
	test_labels = []
	for line in f:
		edge = line.strip('\n\r').split()
		test_pairs.append((int(edge[0]), int(edge[1])))
		test_labels.append(int(edge[2]))
	f.close()
	return test_pairs, test_labels

def remove_edges(G, filename):
    test_pairs, _ = read_test_links(filename)
    edgelist = test_pairs[: len(test_pairs) // 2]
    print('before removing, the # of edges: ', G.number_of_edges())
    G.remove_edges_from(edgelist)
    print('after removing, the # of edges: ', G.number_of_edges())
    return G

# To smooth node features
def smooth(adj, X, gamma):
    # make sure the diag of adj is 1
    for i in range(adj.shape[0]):
        adj[i, i] = 1
    normalized_adj = adj / np.sum(adj, axis=1, keepdims=True)
    smooth_X = normalized_adj.dot(X)
    ada_smooth_X = (1 - gamma) * X + gamma * smooth_X
    return ada_smooth_X






