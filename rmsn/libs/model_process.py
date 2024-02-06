'''
Edited by: Rango
2024.2.2
'''

import numpy as np
import pandas as pd
import os
import logging

import tensorflow as tf
from tensorflow.keras import *

#print time bar
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8,end = "")
    tf.print(timestring)

# Step 1: Construct Model
def create_model(params):
    
    # Data params
    training_data = None if 'training_dataset' not in params else params['training_dataset']
    validation_data = None if 'validation_dataset' not in params else params['validation_dataset']
    test_data = None if 'test_dataset' not in params else params['test_dataset']
    input_size = params['input_size']
    output_size = params['output_size']

    # Network params
    net_name = params['net_name']
    softmax_size = params['softmax_size']
    dropout_rate = params['dropout_rate']
    hidden_layer_size = params['hidden_layer_size']
    memory_activation_type = params['hidden_activation']
    output_activation_type = params['output_activation']
    initial_states = None
    # input layer
    inputs = layers.Input(shape=(None,input_size), dtype=tf.float32)

    # LSTM layer
    lstm = layers.LSTM(hidden_layer_size, activation=memory_activation_type, 
                       return_sequences=True, dropout=dropout_rate)(inputs)

    # flattened_lstm = layers.Flatten()(lstm)

    # Seq2Seq(if need)
    use_seq2seq_feedback = False
    if use_seq2seq_feedback:
        logits = lstm
    else:
        # linear output layer
        logits = layers.Dense(output_size)(lstm)

    # Softmax
    if softmax_size != 0:
        logits_reshaped = layers.Reshape((-1, output_size))(logits)
        core_outputs, softmax_outputs = tf.split(logits_reshaped, [output_size - softmax_size, softmax_size], axis=-1)
        core_activated = layers.Activation(output_activation_type)(core_outputs)
        softmax_activated = layers.Softmax(axis=-1)(softmax_outputs)
        outputs = layers.Concatenate(axis=-1)([core_activated, softmax_activated])
    else:
        outputs = layers.Activation(output_activation_type)(logits)

    # construct model
    model = models.Model(inputs=inputs, outputs=outputs, name=net_name)
    return model

# Step 2: Train Model
# 2.1 Define Loss Function
class CustomLoss(losses.Loss):
    def __init__(self, performance_metric, name="custom_loss"):
        super().__init__(name=name)
        self.performance_metric = performance_metric
        # self.weights = params['weights']
        # self.active_entries = params['active_entries']

    def train_call(self, y_true, y_pred, active_entries, weights):
        if self.performance_metric == "mse":
            loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries * weights) \
                   / tf.reduce_sum(active_entries)
        elif self.performance_metric == "xentropy":
            loss = tf.reduce_sum((y_true * -tf.math.log(y_pred + 1e-8) +
                                  (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
                                  * active_entries * weights) / tf.reduce_sum(active_entries)
        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))
        return loss
    
    def valid_call(self, y_true, y_pred, active_entries):
        if self.performance_metric == "mse":
            loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries ) \
                   / tf.reduce_sum(active_entries)
        elif self.performance_metric == "xentropy":
            loss = tf.reduce_sum((y_true * -tf.math.log(y_pred + 1e-8) +
                                  (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
                                  * active_entries) / tf.reduce_sum(active_entries)
        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"performance_metric": self.performance_metric})
        return config

# 2.2 Define Training Module
class TrainModule(tf.Module):
    def __init__(self, params, name=None):
        super(TrainModule, self).__init__(name=name)
        with self.name_scope:  #相当于with tf.name_scope("demo_module")
            self.epochs = params['num_epochs']
            self.ds_train = params['training_dataset']
            self.ds_valid = params['validation_dataset']
            self.ds_test = params['test_dataset']
            self.optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
            self.loss_func = CustomLoss(params['performance_metric'])

            # train_loss: Track the average loss throughout the training process 
            # by calculating the average of the loss values in all training steps.
            self.train_loss = metrics.Mean(name='train_loss') 
            # train_metric: Track the average MSE throughout the training process
            self.train_metric = metrics.MeanSquaredError(name='train_mse')

            self.valid_loss = metrics.Mean(name='valid_loss')
            self.valid_metric = metrics.MeanSquaredError(name='valid_mse')

    @tf.function
    def train_step(self, model, inputs, outputs, active_entries, weights):

        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = self.loss_func.train_call(outputs, predictions, active_entries, weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
        self.train_loss.update_state(loss)
        self.train_metric.update_state(outputs, predictions)

    @tf.function
    def valid_step(self, model, inputs, outputs, active_entries):

        predictions = model(inputs, training=False)
        loss = self.loss_func.valid_call(outputs, predictions, active_entries)
    
        self.valid_loss.update_state(loss)
        self.valid_metric.update_state(outputs, predictions)

    def train_model(self, model):

        # initialize history
        history = {
            'train_loss': [],
            'train_mse': [],
            'valid_loss': [],
            'valid_mse': []
            }

        for epoch in tf.range(1, self.epochs+1):
        
            for data in self.ds_train:
                weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)
                self.train_step(model, data['inputs'], data['outputs'], data['active_entries'], weights)

            for data in self.ds_valid:
                self.valid_step(model, data['inputs'], data['outputs'], data['active_entries'])

            # save history
            history['train_loss'].append(self.train_loss.result().numpy())
            history['train_mse'].append(self.train_metric.result().numpy())
            history['valid_loss'].append(self.valid_loss.result().numpy())
            history['valid_mse'].append(self.valid_metric.result().numpy())

            # looging and state reset
            logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        
            if epoch%1 ==0:
                printbar()
                tf.print(tf.strings.format(logs,
                (epoch,self.train_loss.result(),self.train_metric.result(),self.valid_loss.result(),self.valid_metric.result())))
                tf.print("")
            
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_metric.reset_states()
            self.valid_metric.reset_states()

        return history

    # Step 3: Evaluate Model
    def evaluate_model(self, model):
        total_loss = 0
        total_metric = 0
        num_batches = 0

        for data in self.ds_test:
            self.valid_step(model, data['inputs'], data['outputs'], data['active_entries'])
            total_loss += self.valid_loss.result().numpy()
            total_metric += self.valid_metric.result().numpy()
            num_batches += 1

            self.valid_loss.reset_states()
            self.valid_metric.reset_states()

        # Calculate the average loss and metrics for the entire dataset
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        return avg_loss, avg_metric # avg_metric=mse

# Step 4: Save and Load Model
# save the hyperparams results to conveniently load model
def add_hyperparameter_results(history,
                               model_folder,
                               net_name,
                               serialisation_name):

    srs = history.copy()
    srs = srs['valid_loss']
    srs = srs.dropna()

    if srs.empty:
        return

    min_loss = srs.min()

    best_idx = list(srs[srs == min_loss].index)[0]

    df = load_hyperparameter_results(model_folder,
                                       net_name)

    df[serialisation_name] = pd.Series({'best_epoch': best_idx,
                                        'validation_loss': min_loss})
    save_name = os.path.join(model_folder, net_name+".csv")
    df.to_csv(save_name)

def load_hyperparameter_results(model_folder,
                               net_name):

    save_name = os.path.join(model_folder, net_name+".csv")
    print(save_name)
    if os.path.exists(save_name):
        return pd.read_csv(save_name, index_col=0)
    else:
        return pd.DataFrame()

def save_model(model, params, history):
    # Saving params
    model_folder = params['model_folder']
    # don't need params['experiment_name']
    relevant_name_parts = [params['net_name'],
                           params['dropout_rate'],
                           params['hidden_layer_size'],
                           params['num_epochs'],
                           params['minibatch_size'],
                           params['learning_rate'],
                           params['max_norm'],
                           params['backprop_length']]

    # save model
    serialisation_name = "_".join([str(s) for s in relevant_name_parts])
    model_path = os.path.join(model_folder, serialisation_name)
    model.save(model_path, save_format = 'tf')
    
    # save history
    history_path = os.path.join(model_folder, "history.csv")
    history.to_csv(history_path, index=False)

    # save hyperparams results
    add_hyperparameter_results(history, model_folder, params['net_name'], serialisation_name)

    logging.info('Model have been saved')

# load model
def get_parameters_from_string(serialisation_string):

    spec = serialisation_string.split("_")
    dropout_rate = float(spec[0])
    hidden_layer_size = int(spec[1])
    num_epochs = int(spec[2])
    minibatch_size = int(spec[3])
    learning_rate = float(spec[4])
    max_norm = float(spec[5])

    return (dropout_rate, hidden_layer_size, num_epochs, minibatch_size, learning_rate, max_norm)

def load_optimal_parameters(net_name, MODEL_ROOT):
    model_folder = os.path.join(MODEL_ROOT, net_name)

    hyperparams_df = load_hyperparameter_results(model_folder, net_name)
    validation_scores = hyperparams_df.loc["validation_loss"]
    # Select optimal
    best_score = validation_scores.min()
    names = np.array(validation_scores.index)
    serialisation_name = names[validation_scores==best_score][0]
    params_string = serialisation_name.replace(net_name+"_", "")
    #print(serialisation_name)
    params = get_parameters_from_string(params_string)
    params = [net_name] + list(params)
    # add serialisation name
    params = params + [serialisation_name]
    return params

def load_model(model_folder, serialisation_name):
    
    model_path = os.path.join(model_folder, serialisation_name)
    model = models.load_model(model_path, compile = False)
    logging.info("Successfully loaded model from {}".format(model_path))

    return model
