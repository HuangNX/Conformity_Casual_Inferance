"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import os
import argparse

#from rmsn.core_routines import train
#import rmsn.core_routines as core
import rmsn.libs.model_process as model_process
import rmsn.libs.data_process as data_process

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
#MODEL_ROOT = configs.MODEL_ROOT
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# EDIT ME! ################################################################################################
# Defines specific parameters to train for - skips hyperparamter optimisation if so
specifications = {
     'rnn_propensity_weighted': (0.1, 4, 100, 64, 0.001, 1.0),
     'treatment_rnn_action_inputs_only': (0.1, 3, 100, 128, 0.01, 2.0),
     'treatment_rnn': (0.1, 4, 100, 64, 0.01, 1.0),
} # decrease learning rate from 0.01 to 0.005 
####################################################################################################################


def rnn_fit(dataset_map, networks_to_train, MODEL_ROOT, b_use_predicted_confounders,
            b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    

    # Get the correct networks to train
    if networks_to_train == "propensity_networks":
        logging.info("Training propensity networks")
        net_names = ['treatment_rnn_action_inputs_only', 'treatment_rnn']
        #net_names = ['treatment_rnn']

    elif networks_to_train == "encoder":
        logging.info("Training R-MSN encoder")
        net_names = ["rnn_propensity_weighted"]

    elif networks_to_train == "user_defined":
        logging.info("Training user defined network")
        raise NotImplementedError("Specify network to use!")

    else:
        raise ValueError("Unrecognised network type")

    logging.info("Running hyperparameter optimisation")

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

    training_data = dataset_map['training_data']
    validation_data = dataset_map['validation_data']
    test_data = dataset_map['test_data']

    # Start Running hyperparam opt
    #opt_params = {}
    for net_name in net_names:

        # Re-run hyperparameter optimisation if parameters are not specified, otherwise train with defined params
        max_hyperparam_runs = 3 if net_name not in specifications else 1

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        use_truncated_bptt = net_name != "rnn_model_bptt" # whether to train with truncated backpropagation through time
        b_propensity_weight = "rnn_propensity_weighted" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name


       # Extract only relevant trajs and shift data
        training_processed = data_process.get_processed_data(training_data, b_predict_actions,
                                                     b_use_actions_only, b_use_predicted_confounders,
                                                     b_use_oracle_confounders, b_remove_x1)
        validation_processed = data_process.get_processed_data(validation_data, b_predict_actions,
                                                       b_use_actions_only, b_use_predicted_confounders,
                                                       b_use_oracle_confounders, b_remove_x1)
        test_processed = data_process.get_processed_data(test_data, b_predict_actions,
                                                 b_use_actions_only, b_use_predicted_confounders,
                                                 b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]
        num_outputs = training_processed['scaled_outputs'].shape[-1]

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
                memory_multiplier = np.random.choice([0.5, 1, 2, 3, 4])
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
                num_epochs = spec[2]
                minibatch_size = spec[3]
                learning_rate = spec[4]
                max_norm = spec[5]
                hidden_activation, output_activation = activation_map[net_name]

            model_folder = os.path.join(MODEL_ROOT, net_name)

            # transform data to tf format
            tf_data_train = data_process.convert_to_tf_dataset(training_processed, minibatch_size)
            tf_data_valid = data_process.convert_to_tf_dataset(validation_processed, minibatch_size)
            tf_data_test = data_process.convert_to_tf_dataset(test_processed, minibatch_size)
            
            #hyperparam_opt = train(net_name, expt_name,
            #                      training_processed, validation_processed, test_processed,
            #                      dropout_rate, memory_multiplier, num_epochs,
            #                      minibatch_size, learning_rate, max_norm,
            #                      use_truncated_bptt,
            #                      num_features, num_outputs, model_folder,
            #                      hidden_activation, output_activation,
            #                      config,
            #                      "hyperparam opt: {} of {}".format(hyperparam_count,
            #                                                        max_hyperparam_runs))

            # construct model parameters
            hidden_layer_size = int(memory_multiplier * num_features)
            model_parameters = {'net_name': net_name,
                    'experiment_name': expt_name,
                    'training_dataset': tf_data_train,
                    'validation_dataset': tf_data_valid,
                    'test_dataset': tf_data_test,
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

            # stpe1: construct model
            tf.keras.backend.clear_session()
            model = model_process.create_model(model_parameters)
            model.summary()

            # step2: train model
            Train = model_process.TrainModule(model_parameters)
            history = Train.train_model(model)
            # history = model_process.train_model(model, model_parameters)
            history = pd.DataFrame(history)

            # step3: save model
            model_process.save_model(model, model_parameters, history)

            # loop control and hyperparameter save
            #hyperparam_count = len(hyperparam_opt.columns)
            hyperparam_count += 1
            if hyperparam_count >= max_hyperparam_runs:
                break

        logging.info("Done")
        #logging.info(hyperparam_opt.T)

        # Flag optimal params
    #logging.info(opt_params)