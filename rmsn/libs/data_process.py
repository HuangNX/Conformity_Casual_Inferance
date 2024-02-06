'''
Edited by: Rango
2024.2.2
'''
import tensorflow as tf
import numpy as np

def get_processed_data(raw_sim_data,
                       b_predict_actions,
                       b_use_actions_only,
                       b_use_predicted_confounders,
                       b_use_oracle_confounders,
                       b_remove_x1,
                       keep_first_point=False):
    """
    Create formatted data to train both propensity networks and seq2seq architecture

    :param raw_sim_data: Data from simulation
    :param scaling_params: means/standard deviations to normalise the data to
    :param b_predict_actions: flag to package data for propensity network to forecast actions
    :param b_use_actions_only:  flag to package data with only action inputs and not covariates
    :param b_predict_censoring: flag to package data to predict censoring locations
    :return: processed data to train specific network
    """
    horizon = 1
    offset = 1

    # Continuous values

    # Binary application
    treatments = raw_sim_data['treatments']
    covariates = raw_sim_data['covariates']
    dataset_outputs = raw_sim_data['outcomes']
    sequence_lengths = raw_sim_data['sequence_length']
    
    if b_use_predicted_confounders:
        predicted_confounders = raw_sim_data['predicted_confounders']

    if b_use_oracle_confounders:
        predicted_confounders = raw_sim_data['confounders']

    num_treatments = treatments.shape[-1]

    # Parcelling INPUTS
    if b_predict_actions:
        if b_use_actions_only:
            inputs = treatments
            inputs = inputs[:, :-offset, :]

            actions = inputs.copy()

        else:
            # Uses current covariate, to remove confounding effects between action and current value
            if (b_use_predicted_confounders):
                print ("Using predicted confounders")
                inputs = np.concatenate([covariates[:, 1:, ], predicted_confounders[:, 1:, ], treatments[:, :-1, ]],
                                        axis=2)
            else:
                inputs = np.concatenate([covariates[:, 1:,], treatments[:, :-1, ]], axis=2)

            actions = inputs[:, :, -num_treatments:].copy()


    else:
        if (b_use_predicted_confounders):
            inputs = np.concatenate([covariates, predicted_confounders, treatments], axis=2)
        else:
            inputs = np.concatenate([covariates, treatments], axis=2)
        
        if not keep_first_point:
            inputs = inputs[:, 1:, :]

        actions = inputs[:, :, -num_treatments:].copy()


    # Parcelling OUTPUTS
    if b_predict_actions:
        outputs = treatments
        outputs = outputs[:, 1:, :]

    else:
        if keep_first_point:
            outputs = dataset_outputs
        else:
            outputs = dataset_outputs[:, 1:, :]


    # Set array alignment
    sequence_lengths = np.array([i - 1 for i in sequence_lengths]) # everything shortens by 1

    # Remove any trajectories that are too short
    inputs = inputs[sequence_lengths > 0, :, :]
    outputs = outputs[sequence_lengths > 0, :, :]
    sequence_lengths = sequence_lengths[sequence_lengths > 0]
    actions = actions[sequence_lengths > 0, :, :]

    # Add active entires
    active_entries = np.zeros(outputs.shape, dtype=np.float32)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])

        if not b_predict_actions:
            for k in range(horizon):
                #include the censoring point too, but ignore future shifts that don't exist
                active_entries[i, :sequence_length-k, k] = 1
        else:
            active_entries[i, :sequence_length, :] = 1

    return {'outputs': outputs,  # already scaled
            'scaled_inputs': inputs,
            'scaled_outputs': outputs,
            'actions': actions,
            'sequence_lengths': sequence_lengths,
            'active_entries': active_entries
            }


def convert_to_tf_dataset(dataset_map, minibatch_size):
    key_map = {'inputs': dataset_map['scaled_inputs'],
               'outputs': dataset_map['scaled_outputs'],
               'active_entries': dataset_map['active_entries'],
               'sequence_lengths': dataset_map['sequence_lengths']}

    if 'propensity_weights' in dataset_map:
        key_map['propensity_weights'] = dataset_map['propensity_weights']

    if 'initial_states' in dataset_map:
        key_map['initial_states'] = dataset_map['initial_states']

    tf_dataset = tf.data.Dataset.from_tensor_slices(key_map)\
                .shuffle(buffer_size = 1000).batch(minibatch_size) \
                .prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset