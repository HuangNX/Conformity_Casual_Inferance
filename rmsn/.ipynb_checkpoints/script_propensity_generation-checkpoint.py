"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import rmsn.configs

from rmsn.core_routine import load_optimal_parameters, propensity_predict
from rmsn.libs.data_process import get_processed_data

import numpy as np
import logging
import os

import tensorflow as tf
import tensorflow_probability as tfp
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

ROOT_FOLDER = rmsn.configs.ROOT_FOLDER
expt_name = "treatment_effects"


def propensity_generation(dataset_map, MODEL_ROOT, b_use_predicted_confounders, b_use_all_data=False,
                          b_use_oracle_confounders=False, b_remove_x1=False):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Load the models（实际上，还是需要参数的）
    action_inputs_only = load_optimal_parameters(net_name='treatment_rnn_action_inputs_only', MODEL_ROOT=MODEL_ROOT)
    action_w_trajectory_inputs = load_optimal_parameters(net_name='treatment_rnn', MODEL_ROOT=MODEL_ROOT)

    # Generate propensity weights for validation data as well - used for MSM which is calibrated on train + valid data
    b_with_validation = False
    # Generate non-stabilised IPTWs (default false)
    b_denominator_only = False

    # Setup tensorflow - setup session to use cpu/gpu
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # set TensorFlow to use all GPU
    #         tf.config.set_visible_devices(gpus, 'GPU')
    #         for gpu in gpus:
    #             # set GPU memery growth
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logging.info("Using GPU with memory growth")
    #     except RuntimeError as e:
    #         # Changing device settings after the program is running may cause errors
    #         logging.info(e)
    # else:
    #     # if no GPU，using CPU
    #     logging.info("No GPU found, using CPU")

    # Config + activation functions
    activation_map = {'rnn_propensity_weighted': ("elu", 'linear'),
                      'rnn_model': ("elu", 'linear'),
                      'rnn_model_bptt': ("elu", 'linear'),
                      'treatment_rnn': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only': ("tanh", 'sigmoid'),
                      'treatment_rnn_softmax': ("tanh", 'sigmoid'),
                      'treatment_rnn_action_inputs_only_softmax': ("tanh", 'sigmoid'),
                      }

    configs = {'action_num': action_inputs_only,
               'action_den': action_w_trajectory_inputs}

    # Setup the simulated datasets
    # rango added 23.11.5###################################
    if b_use_all_data:
        training_data = dataset_map
        validation_data = dataset_map
        test_data = None
    else:
        training_data = dataset_map['training_data']
        validation_data = dataset_map['validation_data']
        test_data = dataset_map['test_data']
    # ######################################################
    # training_data = dataset_map['training_data']
    # validation_data = dataset_map['validation_data']
    # test_data = dataset_map['test_data']

    # Generate propensity weights for validation data if required
    if b_with_validation:
        for k in training_data:
            training_data[k] = np.concatenate([training_data[k], validation_data[k]])

    ##############################################################################################################
    # Functions
    def get_predictions(config):

        net_name = config[0]
        serialisation_name = config[-1]

        hidden_activation, output_activation = activation_map[net_name]

        # Pull datasets
        b_predict_actions = "treatment_rnn" in net_name
        b_use_actions_only = "rnn_action_inputs_only" in net_name

        # Extract only relevant trajs and shift data
        training_processed = get_processed_data(MODEL_ROOT, training_data, b_predict_actions, b_use_actions_only,
                                                     b_use_predicted_confounders, b_use_oracle_confounders, b_remove_x1)
        # rango added 23.10.24
        # if b_with_test:
        #     test_processed = core.get_processed_data(test_data, b_predict_actions, b_use_actions_only,
        #                                              b_use_predicted_confounders, b_use_oracle_confounders, b_remove_x1)

        num_features = training_processed['scaled_inputs'].shape[-1]  # 4 if not b_use_actions_only else 3
        num_timesteps = training_processed['scaled_inputs'].shape[1]
        num_outputs = training_processed['scaled_outputs'].shape[-1]  # 1 if not b_predict_actions else 3  # 5
        #　compute the number of continuous treatments (if networks_to_train == "propensity_networks")
        num_binary_treatments = 0
        for index in range(num_outputs):
            treatment = training_processed['scaled_outputs'][:,:,index]
            if np.all(np.isin(treatment, [0, 1])):
                num_binary_treatments += 1
        num_continuous_treatments = num_outputs - num_binary_treatments

        # Unpack remaining variables
        dropout_rate = config[1]
        hidden_layer_size = config[2]
        memory_multiplier = config[2] / num_features
        num_epochs = config[3]
        minibatch_size = config[4]
        learning_rate = config[5]
        max_norm = config[6]

        model_folder = os.path.join(MODEL_ROOT, net_name)
        model_parameters = {'net_name': net_name,
                    'experiment_name': expt_name,
                    'serialisation_name':serialisation_name,
                    'training_dataset': training_processed,
                    'dropout_rate': dropout_rate,
                    'input_size': num_features,
                    'output_size': num_outputs,
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
       
        multiclass_preds, multiclass_outputs, binary_preds, binary_outputs = propensity_predict(model_parameters)
       
        return multiclass_preds, multiclass_outputs, binary_preds, binary_outputs

    def get_weights(probs, targets):
        targets = targets.astype(np.float64)
        w = probs*targets + (1-probs) * (1-targets)
        return w.prod(axis=2)

    # for multiclass
    def get_probs(preds, outputs, batch_size):
        # Initialize an empty list to collect batch results
        w_list = []
        outputs = tf.squeeze(tf.cast(outputs, tf.int32), axis=-1)
        total_batches, time_steps, class_indices = tf.shape(preds)

        # Process each batch
        for batch_start in range(0, total_batches, batch_size):
            # Calculate batch end, ensuring it does not exceed total_batches
            batch_end = min(batch_start + batch_size, total_batches)
        
            # Slice the preds and outputs for the current batch
            batch_preds = preds[batch_start:batch_end]
            batch_outputs = outputs[batch_start:batch_end]

            # Generate indices for tf.gather_nd
            batch_indices = tf.tile(tf.reshape(tf.range(0, batch_end - batch_start), [-1, 1, 1]), [1, time_steps, 1])
            time_indices = tf.tile(tf.reshape(tf.range(time_steps), [1, -1, 1]), [batch_end - batch_start, 1, 1])
            indices = tf.concat([batch_indices, time_indices, tf.expand_dims(batch_outputs, axis=-1)], axis=-1)
        
            # Gather values for the current batch
            batch_w = tf.gather_nd(batch_preds, indices)
        
            # Append the results for this batch
            w_list.append(batch_w)

        # Concatenate the results from all batches
        w = tf.concat(w_list, axis=0)
        
        #batch_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, time_steps, 1])
        #time_indices = tf.tile(tf.reshape(tf.range(time_steps), [1, -1, 1]), [batch_size, 1, 1])
        ## 合并索引和 output，为 tf.gather_nd 准备
        #indices = tf.concat([batch_indices, time_indices, tf.expand_dims(outputs, axis=-1)], axis=-1)
        
        #w = tf.gather_nd(preds, indices)
        
        return w.numpy()
        
    def get_prob_density(preds, outputs):
        mean_predictions = tf.reduce_mean(preds, axis=0)
        std_dev_predictions = tf.math.reduce_std(preds, axis=0)
        # use normal distribution
        distributions = tfp.distributions.Normal(loc=mean_predictions, scale=std_dev_predictions)
        probs = distributions.prob(outputs)
        probs = probs.numpy()
        return probs

    def get_weights_from_config(config):
        net_name = config[0]
        multiclass_preds, multiclass_outputs, binary_preds, binary_outputs = get_predictions(config)
        # for binary treatments
        binary_w = get_weights(binary_preds, binary_outputs)
        batch_size = config[4]
        multiclass_w = get_probs(multiclass_preds, multiclass_outputs, batch_size)
        print(f"shape of binary_w: {binary_w.shape}, shape of multiclass_w: {multiclass_w.shape}")
        return {'binary_w':binary_w, 'multiclass_w':multiclass_w}

    def get_probabilities_from_config(config):
        net_name = config[0]

        probs, targets = get_predictions(config)

        return probs


    ##############################################################################################################

    # Action with trajs
    weights = {k: get_weights_from_config(configs[k]) for k in configs}

    num_samples = training_data['treatments'].shape[0]
    num_time_steps = training_data['treatments'].shape[1]
    propensity_weights = np.ones([num_samples, num_time_steps-1])
    for key in weights['action_den'].keys():
        den = weights['action_den'][key]
        num = weights['action_num'][key]
        propensity_weights *= 1.0/den if b_denominator_only else num/den

    # only binary treatments
    #den = weights['action_den']
    #num = weights['action_num']
    #propensity_weights = 1.0/den if b_denominator_only else num/den

    # truncation @ 95th and 5th percentiles
    UB = np.percentile(propensity_weights, 95)
    LB = np.percentile(propensity_weights, 5)
    propensity_weights = np.clip(propensity_weights, LB, UB)
    #propensity_weights[propensity_weights > UB] = UB
    #propensity_weights[propensity_weights < LB] = LB

    # Adjust so for 3 trajectories here
    horizon = 1
    (num_patients, num_time_steps) = propensity_weights.shape
    output = np.ones((num_patients, num_time_steps, horizon))

    tmp = np.ones((num_patients, num_time_steps))
    tmp[:, 1:] = propensity_weights[:, :-1]
    propensity_weights = tmp

    for i in range(horizon):
        output[:, :num_time_steps-i, i] = propensity_weights[:, i:]

    propensity_weights = output.cumprod(axis=2)

    suffix = "" if not b_denominator_only else "_den_only"

    if b_with_validation:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores_w_validation{}".format(suffix))
    elif b_use_all_data:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores_w_all{}".format(suffix))
    else:
        save_file = os.path.join(MODEL_ROOT, "propensity_scores{}".format(suffix))

    np.save(save_file, propensity_weights)
    logging.info("Propensity scores saved to {}".format(save_file))


