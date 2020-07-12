import tensorflow as tf
import numpy as np
from graph import load_graph
from utils import node_classification_evaluation, multiclass_classification, get_balance_coefficient, remove_edges, link_prediction_evaluation, read_test_links, get_omega, smooth
import os
import warnings
warnings.filterwarnings('ignore')

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class NAIE(object):
    def __init__(self, config):
        # model parameter
        self.struct = config['struct']
        self.activation = config['activation']

        # train parameter
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.config = config

        self._set_model_parameter()
        self._build_graph()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _set_model_parameter(self):
        struct = self.struct.copy()
        self.W = {}
        self.b = {}

        # encoder parameters
        for i in range(len(struct) - 1):
            self.W['encode_w_' + str(i)] = tf.Variable(tf.random_normal(shape=(struct[i], struct[i + 1]), stddev=1 / np.sqrt(struct[i])), name='encode_w_' + str(i))
            self.b['encode_b_' + str(i)] = tf.Variable(tf.zeros(shape=[struct[i + 1]]), name='encode_b_' + str(i))

        struct.reverse()

        # decoder parameters
        for i in range(len(struct) - 1):
            self.W['decode_w_' + str(i)] = tf.Variable(tf.random_normal(shape=(struct[i], struct[i + 1]), stddev=1 / np.sqrt(struct[i])), name='encode_w_' + str(i))
            self.b['decode_b_' + str(i)] = tf.Variable(tf.zeros(shape=[struct[i + 1]]), name='decode_b_' + str(i))



    def _build_graph(self):
        self.c = tf.placeholder(dtype=tf.float32, shape=[None, self.struct[0]], name='input')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, self.struct[0]], name='target')
        self.adj = tf.placeholder(dtype=tf.float32, shape=[None, None], name='adj')
        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')
 
        struct = self.struct.copy()

        # encode procedure
        h = self.c
        for i in range(len(struct) - 1):
            if self.dropout != 0:
                h = tf.nn.dropout(h, 1 - self.dropout)
            w, b = self.W['encode_w_' + str(i)], self.b['encode_b_' + str(i)]
            h = tf.add(tf.matmul(h, w), b)
            if self.activation != None:
                h = self.activation(h)
        self.h = h

        # decode procedure
        struct.reverse()
        for i in range(len(struct) - 1):
            if self.dropout != 0:
                h = tf.nn.dropout(h, 1 - self.dropout)
            w, b = self.W['decode_w_' + str(i)], self.b['decode_b_' + str(i)]
            h = tf.add(tf.matmul(h, w), b)
            if self.activation != None:
                h = self.activation(h)

        self.reconstruction = h

        # reconstruct loss
        recon_loss = tf.reduce_sum(tf.square(self.reconstruction - self.target))

        # loss of first-order proximity 
        emb = self.h
        emb_square = tf.reduce_sum(tf.square(emb), axis=1, keepdims=True)
        # span the expression of first-order proximity
        L1st_loss = tf.reduce_sum(self.adj * (emb_square + tf.transpose(emb_square) - 2 * tf.matmul(emb, tf.transpose(emb))))

        # regularize loss
        regularized_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.W.values()])

        self.loss = recon_loss + self.alpha * L1st_loss + self.beta * regularized_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    

    def _train_one_batch(self, batch_data):
        _, batch_loss = self.sess.run([self.opt, self.loss], feed_dict={self.c: batch_data.c, 
                                                                      self.target: batch_data.target, 
                                                                      self.adj: batch_data.adj, 
                                                                      self.dropout: self.config['dropout']})
        return batch_loss

    def _batch_data_generater(self, C, target, adj, batch_size):
        indices = [_ for _ in range(C.shape[0])]
        np.random.shuffle(indices)

        for start in np.arange(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            batch_C = C[idx, :]
            batch_target = target[idx, :]
            batch_adj = adj[idx][:, idx]

            yield batch_C, batch_target, batch_adj

    def train(self, data):
        y = data['y']
        C = data['C']
        target = data['target']
        adj = data['adj']
        print(self.epochs)
        for epoch in range(self.epochs):
            loss = 0
            iter = 0
            for batch_C, batch_target, batch_adj in self._batch_data_generater(C, target, adj, self.batch_size):
                batch_data = DotDict()
                batch_data.c = batch_C
                batch_data.target = batch_target
                batch_data.adj = batch_adj
                batch_loss = self._train_one_batch(batch_data)
                loss += batch_loss
                iter += 1

            score = self.evaluate(C, y, task=self.config['task'])
            if self.config['task'] == 'nc':
                print("Epoch {}, Loss={:.5f}, macro={:.5f}".format(epoch, loss / iter, score))
            elif self.config['task'] == 'lp':
                print("Epoch {}, Loss={:.5f}, auc={:.5f}".format(epoch, loss / iter, score))

        final_embedding = self.sess.run(self.h, feed_dict={self.c: C, self.dropout: 0.})

        if self.config['task'] == 'nc':
            macro_avg, micro_avg, macro_var, micro_var = node_classification_evaluation(final_embedding, y)
            print("Classification results: macro-f1={:.3f}+{:7f}, micro-f1={:.3f}+{:.7f}".format(macro_avg, macro_var, micro_avg, micro_var))
            return macro_avg, micro_avg
        elif self.config['task'] == 'lp':
            auc = link_prediction_evaluation(self.config['link_test_pairs'], final_embedding)
            print("Link prediction results: auc={:.5f}".format(auc))
            return auc
            

    def save(self, path):
        self.saver.save(self.sess, path)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def evaluate(self, C, y, task='nc'):
        emb = self.sess.run(self.h, feed_dict={self.c: C, self.dropout: 0})
        if task == 'nc':
            macro, micro = multiclass_classification(emb, y, rnd=2019)
            return macro
        elif task == 'lp':
            auc = link_prediction_evaluation(self.config['link_test_pairs'], emb)
            return auc

def run_NAIE(config):
    graph = load_graph(config['dataset'], labels_is_onehot=False)

    if config['task'] == 'lp':
        graph.G = remove_edges(graph.G, config['lp_test_path'] + config['dataset'] + "_lp_test.edgelist")
        print("Left edges in G: {}".format(graph.G.number_of_edges()))
        test_pairs, test_labels = read_test_links(config['lp_test_path'] + config['dataset'] + "_lp_test.edgelist")
        config['link_test_pairs'] = [(edges[0], edges[1], label) for edges, label in zip(test_pairs, test_labels)]

    y = graph.labels
    X = graph.features
    A = graph.adjcency_matrix(is_sparse=False)
    C = np.concatenate([A, config['lambda'] * X], axis=1)

    smooth_X = smooth(A, X, 1.0)
    smooth_A = smooth(A, A, 1.0)
    if config['strategy'] == 'nc':
        gamma_adj = 1- get_balance_coefficient(graph.G, smooth_A)
        gamma_attr = 1 - get_balance_coefficient(graph.G, smooth_X)
    elif config['strategy'] == 'sw':
        omega = get_omega(graph.G)
        omega = abs(omega)
        if omega > 1:
            omega = 1.0
        gamma_adj = omega
        gamma_attr = omega
    print("gamma_adj={:4f}, gamma_attr={:.4f}".format(gamma_adj, gamma_attr))
    ada_smooth_A = smooth(A, A, gamma_adj)
    ada_smooth_X = smooth(A, X, gamma_attr)
    target = np.concatenate([ada_smooth_A, config['lambda'] * ada_smooth_X], axis=1)

    config['struct'][0] = C.shape[1]
    data = {'C': C, 'target': target, 'adj': A, 'y': y}
    model = NAIE(config)
    model.train(data)

def inference(config):
    graph = load_graph(config['dataset'], labels_is_onehot=False)

    y = graph.labels
    X = graph.features
    A = graph.adjcency_matrix(is_sparse=False)
    C = np.concatenate([A, config['lambda'] * X], axis=1)

    config['struct'][0] = C.shape[1]  
    model = NAIE(config)
    model.restore("./Log/" + config['dataset'] + "/")
    macro, micro = model.evaluate(C, y)
    print("macro={:.4f}, micro={:.4f}".format(macro, micro))




    







        