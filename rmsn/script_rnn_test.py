"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs
from rmsn.configs import load_optimal_parameters

from rmsn.core_routines import test
import rmsn.core_routines as core

import rmsn.libs.model_process as model_process
import rmsn.libs.data_process as data_process

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


# return mse
def rnn_test(dataset_map, MODEL_ROOT, b_use_predicted_confounders, b_use_oracle_confounders=False,
             b_remove_x1=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Setup tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # set TensorFlow to use the first GPU
            gpu0 = gpus[0]
            tf.config.set_visible_devices([gpu0], 'GPU')
            # set GPU memery growth
            tf.config.experimental.set_memory_growth(gpu0, True)
            logging.info("Using GPU with memory growth")
        except RuntimeError as e:
            # Changing device settings after the program is running may cause errors
            logging.info(e)
    else:
        # if no GPU，using CPU
        logging.info("No GPU found, using CPU")

    # change 'rnn_propensity_weighted' to 'treatment_rnn'
    configs = [
        model_process.load_optimal_parameters(net_name = 'rnn_propensity_weighted',MODEL_ROOT = MODEL_ROOT)
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

        training_data = dataset_map['training_data']
        validation_data = dataset_map['validation_data']
        test_data = dataset_map['test_data']

        # scaling_data = pickle_map['scaling_data']  # use scaling data from above

        # Setup some params
        b_predict_actions = "treatment_rnn" in net_name
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "treatment_rnn_action_inputs_only" in net_name

        # In[*]: Compute base MSEs
        # Extract only relevant trajs and shift data
        training_processed = data_process.get_processed_data(training_data, b_predict_actions,
                                                     b_use_actions_only,
                                                     b_use_predicted_confounders, b_use_oracle_confounders, b_remove_x1)
        validation_processed = dataset_map.get_processed_data(validation_data, b_predict_actions,
                                                       b_use_actions_only, b_use_predicted_confounders,
                                                       b_use_oracle_confounders, b_remove_x1)
        test_processed = data_process.get_processed_data(test_data, b_predict_actions,
                                                 b_use_actions_only, b_use_predicted_confounders,
                                                 b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5

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
        model = model_process.load_model(model_folder, serialisation_name)

        #means, output, mse, test_states \
        #    = test(training_processed, validation_processed, test_processed, tf_config,
        #           net_name, expt_name, dropout_rate, num_features, num_outputs,
        #           memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm,
        #           hidden_activation, output_activation, model_folder,
        #           b_use_state_initialisation=False, b_dump_all_states=True)

        #active_entries = test_processed['active_entries']

        #def get_mse_at_follow_up_time(mean, output, active_entires):
        #    mses = np.sum(np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0) \
        #           / active_entires.sum(axis=0).sum(axis=-1)

        #    return pd.Series(mses, index=[idx for idx in range(len(mses))], name=net_name)

        #mse = get_mse_at_follow_up_time(means, output, active_entries)

        projection_map[net_name] = mse
        mse.to_csv(os.path.join(MODEL_ROOT, "results_mse.csv"))
        # save mean estimation
        # save_file = os.path.join(MODEL_ROOT, "results_mean_estimates")
        # np.save(save_file, means)

    return mse
