"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs

from rmsn.core_routine import load_optimal_parameters, effect_predict
from rmsn.libs.data_process import get_processed_data

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import os

import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
RESULTS_FOLDER = rmsn.configs.RESULTS_FOLDER
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# Default params:
expt_name = "treatment_effects"

def rnn_predict(dataset, MODEL_ROOT, b_use_predicted_confounders, b_use_oracle_confounders=False,
             b_remove_x1=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Setup tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # set TensorFlow to use all GPU
            tf.config.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                # set GPU memery growth
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("Using GPU with memory growth")
        except RuntimeError as e:
            # Changing device settings after the program is running may cause errors
            logging.info(e)
    else:
        # if no GPU，using CPU
        logging.info("No GPU found, using CPU")

    configs = [
        load_optimal_parameters(net_name = 'rnn_propensity_weighted',MODEL_ROOT = MODEL_ROOT)
    ]

    # Config
    # change hidden layer of rnn_propensity_weighted to tanh
    activation_map = {'rnn_propensity_weighted': ("tanh", 'linear'),
                      'rnn_propensity_weighted_binary': ("elu", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_actions_only': ("tanh", 'sigmoid')}

    projection_map = {}
    mse_by_followup = {}
    for config in configs:
        net_name = config[0]
        serialisation_name = config[-1]
        projection_map[net_name] = {}

        # scaling_data = pickle_map['scaling_data']  # use scaling data from above

        # Setup some params
        b_predict_actions = "treatment_rnn" in net_name
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "treatment_rnn_action_inputs_only" in net_name

        # Extract only relevant trajs and shift data
        keep_first_point = True
        dataset_processed = get_processed_data(MODEL_ROOT, dataset, b_predict_actions, b_use_actions_only, b_use_predicted_confounders, 
                                                    b_use_oracle_confounders, b_remove_x1, keep_first_point)
        dataset_processed['output_means'] = dataset['output_means']
        dataset_processed['output_stds'] = dataset['output_stds']

        num_features = dataset_processed['scaled_inputs'].shape[-1]  
        num_outputs = dataset_processed['scaled_outputs'].shape[-1]  

        # Pull remaining params
        dropout_rate = config[1]
        hidden_layer_size = config[2]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]
        backprop_length = 60  # we've fixed this
        hidden_activation = activation_map[net_name][0]
        output_activation = activation_map[net_name][1]

        model_folder = os.path.join(MODEL_ROOT, net_name)
        model_parameters = {'net_name': net_name,
                    'experiment_name': expt_name,
                    'serialisation_name':serialisation_name,
                    'dataset': dataset_processed,
                    'dropout_rate': dropout_rate,
                    'input_size': num_features,
                    'output_size': num_outputs,
                    'hidden_layer_size': hidden_layer_size,
                    'num_epochs': num_epochs,
                    'minibatch_size': minibatch_size,
                    'learning_rate': learning_rate,
                    'max_norm': max_norm,
                    'model_folder': model_folder,
                    'hidden_activation': hidden_activation,
                    'output_activation': output_activation,
                    'backprop_length': 60,  # backprop over 60 timesteps for truncated backpropagation through time
                    'softmax_size': 0, #not used in this paper, but allows for categorical actions
                    'performance_metric': 'xentropy' if output_activation == 'sigmoid' else 'mse'}

        predictions = effect_predict(model_parameters)

        # Rescale predictions
        predictions = predictions * dataset_processed['output_stds'] + dataset_processed['output_means']
        #observations = dataset['outcomes']*dataset['output_stds']+dataset['output_means']
        observations = dataset_processed['outputs'] * dataset_processed['output_stds'] + dataset_processed['output_means']

    #return means, ub, lb, observations
    return predictions, observations
