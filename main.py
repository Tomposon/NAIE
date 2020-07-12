from naie import run_NAIE
import tensorflow as tf

config = {
    'strategy':'nc',
    'task': 'nc',
    'lp_test_path': './lp_test_pairs/', 
    'dataset': 'BlogCatalog',
    'lambda': 2,
    'struct': [None, 1000, 500, 128],
    'learning_rate': 0.001,
    'epochs': 50,
    'activation': tf.tanh,
    'alpha': 1,
    'batch_size': 512,
    'beta': 5,
    'dropout':0
}


if __name__ == "__main__":
    run_NAIE(config)
    