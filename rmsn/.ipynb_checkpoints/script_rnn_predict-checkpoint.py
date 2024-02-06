"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs
from rmsn.configs import load_optimal_parameters

from rmsn.core_routines import predict
import rmsn.core_routines as core

from rmsn.libs.model_rnn import RnnModel
import rmsn.libs.net_helpers as helpers

from sklearn.metrics import roc_auc_score, average_precision_score

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
    tf_device = 'gpu'
    if tf_device == "cpu":
        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
    else:
        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'GPU': 1})
        tf_config.gpu_options.allow_growth = True

    configs = [
        load_optimal_parameters('rnn_propensity_weighted',
                                expt_name, MODEL_ROOT,
                                add_net_name=True)
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

        projection_map[net_name] = {}

        # scaling_data = pickle_map['scaling_data']  # use scaling data from above

        # Setup some params
        b_predict_actions = "treatment_rnn" in net_name
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "treatment_rnn_action_inputs_only" in net_name

        # In[*]: Compute base MSEs
        # Extract only relevant trajs and shift data
        keep_first_point = True
        dataset_processed = core.get_processed_data(dataset, b_predict_actions, b_use_actions_only, b_use_predicted_confounders, 
                                                    b_use_oracle_confounders, b_remove_x1, keep_first_point)
        dataset_processed['output_means'] = dataset['output_means']
        dataset_processed['output_stds'] = dataset['output_stds']

        num_features = dataset_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = dataset_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

        # Pull remaining params
        dropout_rate = config[1]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]
        backprop_length = 60  # we've fixed this
        hidden_activation = activation_map[net_name][0]
        output_activation = activation_map[net_name][1]

        # Run tests
        model_folder = os.path.join(MODEL_ROOT, net_name)

        means, ub, lb, test_states \
            = predict(dataset_processed, tf_config,
                   net_name, expt_name, dropout_rate, num_features, num_outputs,
                   memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
                   hidden_activation, output_activation, model_folder,
                   b_use_state_initialisation=False, b_dump_all_states=True)
        
        #observations = dataset['outcomes']*dataset['output_stds']+dataset['output_means']
        observations = dataset_processed['outputs']

    return means, ub, lb, observations
