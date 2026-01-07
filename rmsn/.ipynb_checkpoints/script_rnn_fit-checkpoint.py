"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

from re import I
import rmsn.configs

import tensorflow as tf
from tensorflow.keras import *
import numpy as np
import pandas as pd
import logging
import os
import shutil
import argparse
from rmsn.core_routine import propensity_model_train, predictive_model_train, model_evaluate
from rmsn.libs.data_process import get_processed_data

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
#MODEL_ROOT = configs.MODEL_ROOT
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# EDIT ME! ################################################################################################
# Defines specific parameters to train for - skips hyperparamter optimisation if so
# (dropout_rate, memory_multiplier, num_epochs, minibatch_size, learning_rate, max_norm)
# base model
specifications = {
'treatment_rnn_action_inputs_only': (0.1, 6, 200, 64, 0.005, 0.5),
'treatment_rnn': (0.1, 3, 200, 64, 0.01, 0.5),
'rnn_propensity_weighted': (0.1, 3, 200, 64, 0.01, 0.5),
} 
# add model
# specifications = {
#     'treatment_rnn_action_inputs_only': (0.2, 4, 200, 128, 0.005, 2.0),
#     'treatment_rnn': (0.1, 3, 200, 64, 0.01, 0.5),
#     'rnn_propensity_weighted': (0.1, 6, 200, 64, 0.01, 2.0),
# }

# default
# specifications = {
#     'treatment_rnn_action_inputs_only': (0.1, 5, 100, 128, 0.005, 2.0),
#     'treatment_rnn': (0.1, 5, 100, 64, 0.005, 1.0),
#     'rnn_propensity_weighted': (0.1, 5, 100, 64, 0.005, 1.0),
# }
####################################################################################################################


def rnn_fit(dataset_map, networks_to_train, MODEL_ROOT, b_use_predicted_confounders,
            b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    

    # Get the correct networks to train
    if networks_to_train == "propensity_networks":
        logging.info("Training propensity networks")
        # net_names = ['treatment_rnn_action_inputs_only']
        net_names = ['treatment_rnn']

    elif networks_to_train == "encoder":
        logging.info("Training R-MSN encoder")
        net_names = ["rnn_propensity_weighted"]

    elif networks_to_train == "user_defined":
        logging.info("Training user defined network")
        raise NotImplementedError("Specify network to use!")

    else:
        raise ValueError("Unrecognised network type")


    # Experiment name
    expt_name = "treatment_effects"

    # Possible networks to use along with their activation functions
    # change hidden layer of rnn_propensity_weighted to tanh
    activation_map = {'rnn_propensity_weighted': ("tanh", 'linear'),
                      'rnn_propensity_weighted_logistic': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid')
                      }


    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    # Start Running hyperparam opt
    #opt_params = {}
    mse_dict = {}
    for net_name in net_names:

        # Re-run hyperparameter optimisation if parameters are not specified, otherwise train with defined params
        max_hyperparam_runs = 25 if net_name not in specifications else 1
        if max_hyperparam_runs > 1:
            logging.info("Running hyperparameter optimisation for {}".format(net_name))

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        use_truncated_bptt = net_name != "rnn_model_bptt" # whether to train with truncated backpropagation through time
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name


       # Extract only relevant trajs and shift data
        training_processed = get_processed_data(MODEL_ROOT, training_data, b_predict_actions,
                                                     b_use_actions_only, b_use_predicted_confounders,
                                                     b_use_oracle_confounders, b_remove_x1)
        validation_processed = get_processed_data(MODEL_ROOT, validation_data, b_predict_actions,
                                                       b_use_actions_only, b_use_predicted_confounders,
                                                       b_use_oracle_confounders, b_remove_x1)
        test_processed = get_processed_data(MODEL_ROOT, test_data, b_predict_actions,
                                                 b_use_actions_only, b_use_predicted_confounders,
                                                 b_use_oracle_confounders, b_remove_x1)


        num_features = training_processed['scaled_inputs'].shape[-1]
        num_timesteps = training_processed['scaled_inputs'].shape[1]
        num_outputs = training_processed['scaled_outputs'].shape[-1]

        #　compute the number of continuous treatments (if networks_to_train == "propensity_networks")
        if networks_to_train == "propensity_networks":
            num_binary_treatments = 0
            for index in range(num_outputs):
                treatment = training_processed['scaled_outputs'][:,:,index]
                if np.all(np.isin(treatment, [0, 1])):
                    num_binary_treatments += 1
        else:
            num_binary_treatments = num_outputs
        num_continuous_treatments = num_outputs - num_binary_treatments

        # Load propensity weights if they exist
        if b_propensity_weight:

            if net_name == 'rnn_propensity_weighted_den_only':
                # use un-stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores_den_only.npy"))
            elif net_name == "rnn_propensity_weighted_logistic":
                # Use logistic regression weights
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))
                tmp = np.load(os.path.join(MODEL_ROOT, "propensity_scores_logistic.npy"))
                propensity_weights = tmp[:propensity_weights.shape[0], :, :]
            else:
                # use stabilised IPTWs generated by propensity networks
                propensity_weights = np.load(os.path.join(MODEL_ROOT, "propensity_scores.npy"))

            logging.info("Net name = {}. Mean-adjusting!".format(net_name))

            propensity_weights /= propensity_weights.mean()

            training_processed['propensity_weights'] = np.array(propensity_weights, dtype='float32')

        # Start hyperparamter optimisation (training model directly)
        hyperparam_count = 0
        while True:

            if net_name not in specifications:

                dropout_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
                memory_multiplier = np.random.choice([3, 4, 5, 6, 7])
                num_epochs = 100
                minibatch_size = np.random.choice([64, 128, 256])
                learning_rate = np.random.choice([0.01, 0.005, 0.001])  #([0.01, 0.001, 0.0001])
                max_norm = np.random.choice([0.5, 1.0, 2.0, 4.0])
                hidden_activation, output_activation = activation_map[net_name]

            else:
                spec = specifications[net_name]
                logging.info("Using specifications for {}: {}".format(net_name, spec))
                dropout_rate = spec[0]
                memory_multiplier = spec[1]
                #num_epochs = 10
                num_epochs = spec[2]
                minibatch_size = spec[3]
                learning_rate = spec[4]
                max_norm = spec[5]
                hidden_activation, output_activation = activation_map[net_name]

            model_folder = os.path.join(MODEL_ROOT, net_name)
            if os.path.exists(model_folder):
               # Need to delete previously saved model.
                shutil.rmtree(model_folder)
                os.mkdir(model_folder)
                print("Directory ", model_folder, " Created ")

            # construct model parameters
            num_input_shape = num_features + 19 * num_continuous_treatments
            hidden_layer_size = int(memory_multiplier * num_input_shape)
            model_parameters = {'net_name': net_name,
                    'experiment_name': expt_name,
                    'training_dataset': training_processed,
                    'validation_dataset': validation_processed,
                    'test_dataset': test_processed,
                    'dropout_rate': dropout_rate,
                    'input_size': num_features,
                    'output_size': num_outputs,
                    'treatment_only': b_use_actions_only,
                    'time_steps': num_timesteps,
                    'hidden_layer_size': hidden_layer_size,
                    'num_epochs': num_epochs,
                    'minibatch_size': minibatch_size,
                    'learning_rate': learning_rate,
                    'max_norm': max_norm,
                    'model_folder': model_folder,
                    'hidden_activation': hidden_activation,
                    'output_activation': output_activation,
                    'backprop_length': 60,  # backprop over 60 timesteps for truncated backpropagation through time
                    'num_continuous': num_continuous_treatments, # number of continuous treatments
                    'softmax_size': 20 * num_continuous_treatments, #equals to bins * num_continuous
                    'predict_size': num_outputs if num_continuous_treatments == 0 else num_outputs + 19 * num_continuous_treatments,
                    'performance_metric': 'xentropy' if output_activation == 'sigmoid' else 'mse'}

            if net_name == "rnn_propensity_weighted":
                history = predictive_model_train(model_parameters)
            elif "treatment_rnn" in net_name:
                history = propensity_model_train(model_parameters)
            else:
                raise ValueError("Unrecognised network type to train")

            hyperparam_count += 1
            if hyperparam_count >= max_hyperparam_runs:
                break

        # evaluate model if net_names = ["rnn_propensity_weighted"]
        if net_name == "rnn_propensity_weighted":
            rmse = model_evaluate(model_parameters)
            print('RMSE after restoring the saved model without strategy: {}'.format(rmse))