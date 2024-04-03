'''
Edited by: Rango
2024.2.2
'''
from random import sample
import numpy as np
import pandas as pd
import os
import logging

import tensorflow as tf
from tensorflow.keras import *
#import tensorflow_probability as tfp

# Step 1: Construct Model
def create_model(params):
    # Data params
    input_size = params['input_size']
    output_size = params['output_size']
    predict_size = params['predict_size'] # predict_size = output_size if softmax_size = 0 else output_size - num_continuous + softmax_size
    treatment_only = params['treatment_only']
    continuous_size = params['num_continuous']
    mc_size = 0

    # Network params
    net_name = params['net_name']
    softmax_size = params['softmax_size']  # equals to bins number
    dropout_rate = params['dropout_rate']
    hidden_layer_size = params['hidden_layer_size']
    memory_activation_type = params['hidden_activation']
    output_activation_type = params['output_activation']
    #initial_states = None
    # input layer
    inputs = layers.Input(shape=(None,input_size), dtype=tf.float32)
    # define initial states 
    initial_h =layers.Input(shape=(hidden_layer_size,), dtype=tf.float32, name='initial_h')
    initial_c =layers.Input(shape=(hidden_layer_size,), dtype=tf.float32, name='initial_c')

    ## Discretization layer (only for propensity network)
    #if softmax_size != 0:
    #    if treatment_only:
    #        # Using tf.split to split continuous and binary inputs
    #        continuous_inputs, binary_inputs = tf.split(inputs, [continuous_size, output_size - continuous_size], axis=-1)
    #        # Apply Discretization on continuous inputs
    #        discretized_continuous = layers.Discretization(num_bins=softmax_size)(continuous_inputs)
    #        # Cast discretized continuous inputs to float32 to match binary inputs
    #        discretized_continuous = tf.cast(discretized_continuous, tf.float32)
    #        # Concatenate discretized continuous inputs with binary inputs
    #        process_inputs = layers.Concatenate(axis=-1)([discretized_continuous, binary_inputs])
    #    else:
    #        non_treatment_inputs, treatment_inputs = tf.split(inputs, [input_size - output_size, output_size], axis=-1)
    #        continuous_treatment, binary_treatment = tf.split(treatment_inputs, [continuous_size, output_size - continuous_size], axis=-1)
    #        discretized_continuous = layers.Discretization(num_bins=20)(continuous_treatment)
    #        discretized_continuous = tf.cast(discretized_continuous, tf.float32)
    #        process_inputs = layers.Concatenate(axis=-1)([non_treatment_inputs, discretized_continuous, binary_treatment])
    #else:
    #    process_inputs = inputs

    # LSTM layer
    lstm, state_h, state_c = layers.LSTM(hidden_layer_size, activation=memory_activation_type, 
                       return_sequences=True, return_state=True, dropout=dropout_rate)(inputs, initial_state=[initial_h, initial_c])

    # flattened_lstm = layers.Flatten()(lstm)

    # linear output layer
    logits = layers.Dense(predict_size)(lstm)

    # Softmax（对应多分类任务）
    if softmax_size != 0:
        logits_reshaped = layers.Reshape((-1, predict_size))(logits)
        softmax_outputs, core_outputs = tf.split(logits_reshaped, [softmax_size, predict_size - softmax_size], axis=-1)
        core_activated = layers.Activation(output_activation_type)(core_outputs)
        softmax_activated = layers.Softmax(axis=-1)(softmax_outputs)
        outputs = layers.Concatenate(axis=-1)([softmax_activated, core_activated])
    
    # MC dropout（对应连续任务）
    elif mc_size != 0:
        logits_reshaped = layers.Reshape((-1, output_size))(logits)
        mc_outputs, core_outputs = tf.split(logits_reshaped, [mc_size, output_size - mc_size], axis=-1)
        mc_activated = layers.Dense(1)(mc_outputs)
        mc_activated = layers.Dropout(dropout_rate)(mc_activated, training=True) # 启用dropout，即使在推断时
        core_activated = layers.Activation(output_activation_type)(core_outputs)
        outputs = layers.Concatenate(axis=-1)([mc_activated, core_activated])
    
    else:
        outputs = layers.Activation(output_activation_type)(logits)

    # construct model
    model = models.Model(inputs=[inputs, initial_h, initial_c], outputs=[outputs, state_h, state_c], name=net_name)
    return model

# Step 2: Train Model
# 2.1 Define Loss Function 
class CustomLoss(losses.Loss):
    def __init__(self, params, name="custom_loss"):
        super().__init__(name=name) #reduction=losses.Reduction.NONE
        self.performance_metric = params['performance_metric']
        self.num_gpus = params['num_gpus']
        self.global_batch_size = params['global_batch_size']
        #self.softmax_size = params['softmax_size']

    # Need Modified: if softmax!=0 ...
    def train_call(self, y_true, y_pred, active_entries, weights):
        active_timesteps_per_sample = tf.cast(tf.reduce_any(active_entries > 0, axis=-1), tf.float32)
        weights = tf.constant(1.0) if weights is None else tf.squeeze(weights, axis=-1)
        sample_weights = active_timesteps_per_sample * weights
        if self.performance_metric == "mse":
            mse = losses.MeanSquaredError(reduction=losses.Reduction.NONE)
            per_example_loss = mse(y_true, y_pred, sample_weight = sample_weights)
            #loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries * weights) \
            #        / tf.reduce_sum(active_entries)
            #per_example_loss = (tf.square(y_true - y_pred) * active_entries * weights) \
            #                    / tf.reduce_sum(active_entries)
        elif self.performance_metric == "xentropy":
            bce = losses.BinaryCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)
            per_example_loss = bce(y_true, y_pred, sample_weight = sample_weights)
            #loss = tf.reduce_sum((y_true * -tf.math.log(y_pred + 1e-8) +
            #                        (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
            #                        * active_entries * weights) / tf.reduce_sum(active_entries)
            #per_example_loss = ((y_true * -tf.math.log(y_pred + 1e-8) + \
            #                    (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8)) * active_entries * weights) / tf.reduce_sum(active_entries)
        elif self.performance_metric == "sparse_xentropy":
            scce = losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)
            per_example_loss = scce(y_true, y_pred, sample_weight = sample_weights)
        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))

        # 将总和除以gpu数，获得全局平均损失
        #return loss * (1./self.num_gpus)
        # average loss in individual
        per_example_loss = tf.reduce_mean(per_example_loss, axis=-1)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def valid_call(self, y_true, y_pred):
        if self.performance_metric == "mse":
            #loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries ) \
            #        / tf.reduce_sum(active_entries)
            #loss = tf.square(y_true - y_pred)
            mse = losses.MeanSquaredError(reduction=losses.Reduction.NONE)
            loss = mse(y_true, y_pred)

        elif self.performance_metric == "xentropy":
            #loss = (y_true * -tf.math.log(y_pred + 1e-8) +
            #        (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
            bce = losses.BinaryCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)
            loss = bce(y_true, y_pred)
        
        elif self.performance_metric == "sparse_xentropy":
            scce = losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)
            loss = scce(y_true, y_pred)

        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"performance_metric": self.performance_metric, "global_batch_size": self.global_batch_size})
        return config


    # Step 3: Evaluate Model
    def eval_step(self, new_model):
        new_optimizer = optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        total_loss = 0
        total_metric = 0
        num_batches = 0

        for data in self.ds_test:
            self.distributed_valid_step(new_model, data['inputs'], data['outputs'], data['active_entries'])
            total_loss += self.valid_loss.result().numpy()
            total_metric += self.valid_metric.result().numpy()
            num_batches += 1

            self.valid_loss.reset_states()
            self.valid_metric.reset_states()

        # Calculate the average loss and metrics for the entire dataset
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        return avg_loss, avg_metric # avg_metric=mse


# Step 4: Use Model to Make Predictions
#def get_predictions(model, dataset, pred_times=500):
    
#    # setup predictions
#    all_predictions = []
#    for data_chunk in dataset:
#        chunk_predictions = []
#        for _ in range (pred_times):
#            predictions = [model.predict(data_chunk['inputs'], verbose=False) for _ in range(pred_times)]
#            predictions.append(predictions)

#        predictions = np.stack(predictions)
#        chunk_predictions.append(predictions)
#        #predictions = tf.stack(predictions, axis=0)

#    # Dumping output
#    predictions = tf.concat(predictions, axis=0)
#    mean_estimate = tf.reduce_mean(predictions, axis=0)
#    upper_bound = tfp.stats.percentile(predictions, q=95.0, axis=0)
#    lower_bound = tfp.stats.percentile(predictions, q=5.0, axis=0)

#    return {'mean_pred': mean_estimate, 'upper_bound': upper_bound, 'lower_bound': lower_bound}


