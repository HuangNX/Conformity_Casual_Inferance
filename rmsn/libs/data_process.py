'''
Edited by: Rango
2024.2.2
'''
import tensorflow as tf
import numpy as np
import logging
import os
from tensorflow.keras.layers import Discretization, Input
from tensorflow.keras.models import Model, load_model

def discretize_treatment(MODEL_ROOT, treatments, sample_size = 100000):
    logging.info("Treatment Discretization ...")
    model_path = os.path.join(MODEL_ROOT, 'discretization_model')
    continuous_indices = [index for index in range(treatments.shape[2]) if not np.all(np.isin(treatments[:, :, index], [0, 1]))]

    if os.path.exists(model_path):
    # 加载已存在的模型
        logging.info("Loading existing discretization model from: %s", model_path)
        discretization_model = load_model(model_path)
    else:
        logging.info("Discretization model not found, creating a new one.")
        # 初始化离散化层字典，用于存储每个连续特征的离散化层
        inputs = []
        outputs = []
        # 为每个连续特征创建并适配离散化层
        for index in continuous_indices:
            # 提取当前连续特征
            continuous_feature = treatments[:, :, index].flatten()  # 将其展平以适配Discretization层
            if len(continuous_feature) <= sample_size:
                sampled_feature = continuous_feature
            else:
                # 随机选择sample_size个数据点进行采样
                sampled_indices = np.random.choice(len(continuous_feature), size=sample_size, replace=False)
                sampled_feature = continuous_feature[sampled_indices]
    
            # 为每个离散化层创建一个独立的输入
            input_layer = Input(shape=(1,), dtype=tf.float32)
            inputs.append(input_layer)

            # 创建并适配Discretization层
            discretization_layer = Discretization(num_bins=20)  # 假设你想要分成20个桶
            discretization_layer.adapt(sampled_feature.reshape(-1, 1))  # 需要二维数组
    
            # 将Discretization层作为模型的一部分
            output_layer = discretization_layer(input_layer)
            outputs.append(output_layer)
        
        discretization_model = Model(inputs=inputs, outputs=outputs)
        discretization_model.save(model_path)
        logging.info("Saved discretization model to: %s", model_path)

    # 应用离散化层到连续特征
    inputs = [treatments[:, :, idx].reshape(-1, 1) for idx in continuous_indices]
    bin_edges = []
    for layer in discretization_model.layers:
        if isinstance(layer, tf.keras.layers.Discretization):
            # 直接访问 bin_boundaries 属性
            bin_boundaries = layer.bin_boundaries
            # 因为 bin_boundaries 是 ListWrapper，我们将其转换为列表，然后转换为 NumPy 数组
            bin_boundaries_np = np.array(list(bin_boundaries))
            bin_edges.append(bin_boundaries_np)

    for idx in continuous_indices:
        continuous_feature = treatments[:, :, idx].flatten().reshape(-1, 1)
        discretized_feature = np.digitize(continuous_feature, bin_edges[idx])
        treatments[:, :, idx] = discretized_feature.reshape(treatments.shape[0], treatments.shape[1])

    return treatments

def get_processed_data(MODEL_ROOT,
                       raw_sim_data,
                       b_predict_actions,
                       b_use_actions_only,
                       b_use_predicted_confounders,
                       b_use_oracle_confounders,
                       b_remove_x1,
                       keep_first_point=False,
                       discretization_layers = None):
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

        treatments = discretize_treatment(MODEL_ROOT, treatments)

        if b_use_actions_only:
            inputs = treatments
            inputs = inputs[:, :-offset, :]

            actions = inputs.copy()

        else:
            # Uses current covariate, to remove confounding effects between action and current value
            if (b_use_predicted_confounders):
                #print ("Using predicted confounders")
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

def data_generator(dataset_map, batch_size):
    def generator():
        num_samples = len(dataset_map['scaled_inputs'])
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            
            # 根据当前批次的索引构建key_map
            key_map = {
                'inputs': dataset_map['scaled_inputs'][start_idx:end_idx],
                'outputs': dataset_map['scaled_outputs'][start_idx:end_idx],
                'active_entries': dataset_map['active_entries'][start_idx:end_idx],
                'sequence_lengths': dataset_map['sequence_lengths'][start_idx:end_idx]
            }

            # 根据数据集中是否存在额外的键来动态添加这些键
            if 'propensity_weights' in dataset_map:
                key_map['propensity_weights'] = dataset_map['propensity_weights'][start_idx:end_idx]

            if 'initial_states' in dataset_map:
                key_map['initial_states'] = dataset_map['initial_states'][start_idx:end_idx]

            yield key_map

    return generator

def convert_to_tf_dataset(dataset_map, minibatch_size, for_train = True):
    key_map = {'inputs': dataset_map['scaled_inputs'],
               'outputs': dataset_map['scaled_outputs'],
               'active_entries': dataset_map['active_entries'],
               'sequence_lengths': dataset_map['sequence_lengths']}

    if 'propensity_weights' in dataset_map:
        key_map['propensity_weights'] = dataset_map['propensity_weights']

    if 'initial_states' in dataset_map:
        key_map['initial_states'] = dataset_map['initial_states']

    if for_train:
        tf_dataset = tf.data.Dataset.from_tensor_slices(key_map)\
                    .shuffle(buffer_size = 1000).batch(minibatch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices(key_map)\
                    .batch(minibatch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset

def convert_to_tf_dataset_via_generator(dataset_map, batch_size, for_train=True):

    output_types = {
        'inputs': tf.float32,  
        'outputs': tf.float32,
        'active_entries': tf.float32,
        'sequence_lengths': tf.int32,
    }

    output_shapes = {
        'inputs': tf.TensorShape([None, dataset_map['scaled_inputs'].shape[1], dataset_map['scaled_inputs'].shape[-1]]),  
        'outputs': tf.TensorShape([None, dataset_map['scaled_outputs'].shape[1], dataset_map['scaled_outputs'].shape[-1]]),
        'active_entries': tf.TensorShape([None, dataset_map['active_entries'].shape[1], dataset_map['active_entries'].shape[-1]]),
        'sequence_lengths': tf.TensorShape([None]),
    }

    # 如果 dataset_map 包含额外的键，也需要在这里添加
    if 'propensity_weights' in dataset_map:
        output_types['propensity_weights'] = tf.float32
        output_shapes['propensity_weights'] = tf.TensorShape([None, dataset_map['propensity_weights'].shape[1], dataset_map['propensity_weights'].shape[-1]])


    if 'initial_states' in dataset_map:
        output_types['initial_states'] = tf.float32
        output_shapes['initial_states'] = tf.TensorShape([None, dataset_map['initial_states'].shape[1], dataset_map['initial_states'].shape[-1]])


    dataset = tf.data.Dataset.from_generator(
        generator=data_generator(dataset_map, batch_size),
        output_types=output_types,
        output_shapes=output_shapes
    )

    if for_train:
        dataset = dataset.shuffle(buffer_size=1000).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset