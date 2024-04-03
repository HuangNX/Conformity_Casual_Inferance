'''
Edited by: Rango
2024.2.2
'''

import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import *
import tensorflow_probability as tfp

# strategy = tf.distribute.MirroredStrategy()
# NAN debug
#tf.debugging.enable_check_numerics()
# 全局启用/停用eager模式
#tf.config.run_functions_eagerly(True)

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
    #training_data = None if 'training_dataset' not in params else params['training_dataset']
    #validation_data = None if 'validation_dataset' not in params else params['validation_dataset']
    #test_data = None if 'test_dataset' not in params else params['test_dataset']
    input_size = params['input_size']
    output_size = params['output_size']

    # Network params
    net_name = params['net_name']
    softmax_size = params['softmax_size']
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

    # LSTM layer
    lstm, state_h, state_c = layers.LSTM(hidden_layer_size, activation=memory_activation_type, 
                       return_sequences=True, return_state=True, dropout=dropout_rate)(inputs, initial_state=[initial_h, initial_c])

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
    model = models.Model(inputs=[inputs, initial_h, initial_c], outputs=[outputs, state_h, state_c], name=net_name)
    return model

# Step 2: Train Model
# 2.1 Define Loss Function
# with strategy.scope():
class CustomLoss(losses.Loss):
    def __init__(self, performance_metric, num_gpus, name="custom_loss"):
        super().__init__(name=name) #reduction=losses.Reduction.NONE
        self.performance_metric = performance_metric
        self.num_gpus = num_gpus
        # self.weights = params['weights']
        # self.active_entries = params['active_entries']

    def train_call(self, y_true, y_pred, active_entries, weights):
        if self.performance_metric == "mse":
            loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries * weights) \
                    / tf.reduce_sum(active_entries)
            #per_example_loss = (tf.square(y_true - y_pred) * active_entries * weights) \
            #                    / tf.reduce_sum(active_entries)
        elif self.performance_metric == "xentropy":
            loss = tf.reduce_sum((y_true * -tf.math.log(y_pred + 1e-8) +
                                    (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))
                                    * active_entries * weights) / tf.reduce_sum(active_entries)
            #per_example_loss = ((y_true * -tf.math.log(y_pred + 1e-8) + \
            #                   (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8)) * active_entries * weights) / tf.reduce_sum(active_entries)

        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))

        # 将总和除以gpu数，获得全局平均损失
        return loss * (1./self.num_gpus)
        #return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def valid_call(self, y_true, y_pred):
        if self.performance_metric == "mse":
           #loss = tf.reduce_sum(tf.square(y_true - y_pred) * active_entries ) \
           #        / tf.reduce_sum(active_entries)
            loss = tf.square(y_true - y_pred)

        elif self.performance_metric == "xentropy":
            loss = (y_true * -tf.math.log(y_pred + 1e-8) +
                   (1 - y_true) * -tf.math.log(1 - y_pred + 1e-8))

        else:
            raise ValueError("Unknown performance metric {}".format(self.performance_metric))

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"performance_metric": self.performance_metric, "global_batch_size": self.global_batch_size})
        return config

# 2.2 Define Training Module
class TrainModule(tf.Module):
    def __init__(self, params, strategy, name=None):
        super(TrainModule, self).__init__(name=name)
        self.strategy = strategy
        with self.name_scope:  #相当于with tf.name_scope("demo_module")
            self.epochs = params['num_epochs']
            self.ds_train = params['training_dataset']
            self.ds_valid = params['validation_dataset']
            self.ds_test = params['test_dataset']
            self.input_size = params['input_size']
            self.minibatch_size = params['minibatch_size']
            self.global_batch_size = params['global_batch_size']
            self.hidden_layer_size = params['hidden_layer_size']
            self.performance_metric = params['performance_metric']
            self.max_global_norm = params['max_norm']
            self.backprop_length = params['backprop_length']
            self.model_folder = params['model_folder']
            self.optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
            self.loss_func = CustomLoss(self.performance_metric, int(self.global_batch_size / self.minibatch_size))

            # Set up checkpoint directory
            self.checkpoint_dir = os.path.join(self.model_folder, 'training_checkpoints')

            # train_loss: Track the average loss throughout the training process 
            # by calculating the average of the loss values in all training steps.
            #self.train_loss = metrics.Mean(name='train_loss') 
            # train_metric: Track the average MSE throughout the training process
            self.train_metric = metrics.MeanSquaredError(name='train_mse')

            self.valid_loss = metrics.Mean(name='valid_loss')
            self.valid_metric = metrics.MeanSquaredError(name='valid_mse')

    def train_step(self, model, inputs, outputs, active_entries, weights, chunk_sizes):
        
        with tf.GradientTape() as tape:
            #segment_predictions = []
            #start = 0
            #states = None
            #for i, chunk_size in enumerate(chunk_sizes):
            #    input_chunk = tf.slice(inputs, [0, start, 0], [-1, chunk_size, self.input_size])
            #    if states is None:
            #        batch_size = tf.shape(input_chunk)[0]
            #        initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            #        segment_prediction, state_h, state_c = model([input_chunk, initial_state, initial_state], training=True)
            #    else:
            #        segment_prediction, state_h, state_c = model([input_chunk, states[0], states[1]], training=True)
           
            #    segment_predictions.append(segment_prediction)
            #    # break links between states for truncated bptt
            #    states = [state_h, state_c]
            #    #states = tf.identity(states)
            #    # Starting point
            #    start += chunk_size

            ## Dumping output
            #predictions = tf.concat(segment_predictions, axis=1)
            batch_size = tf.shape(inputs)[0]
            initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
            predictions,_,_ = model([inputs,initial_state, initial_state], training=True)
            # Compute loss
            loss = self.loss_func.train_call(outputs, predictions, active_entries, weights)
        
            #predictions = model(inputs, training=True)
            #loss = self.loss_func.train_call(outputs, predictions, active_entries, weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = self.max_global_norm)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        #self.train_loss.update_state(loss)
        self.train_metric.update_state(outputs, predictions)

        return loss, predictions, outputs

    @tf.function
    def distributed_train_step(self, model, inputs, outputs, active_entries, weights, chunk_sizes):
        per_replica_losses, predictions, outputs = self.strategy.run(self.train_step, args=(model, inputs, outputs, active_entries, weights, chunk_sizes))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), predictions, outputs

    #@tf.function
    def valid_step(self, model, inputs, outputs, active_entries):

        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros([batch_size, self.hidden_layer_size], dtype=tf.float32)
        predictions, _, _ = model([inputs, initial_state, initial_state], training=False)
        loss = self.loss_func.valid_call(outputs, predictions)
        sample_weight = active_entries/tf.reduce_sum(active_entries)
    
        self.valid_loss.update_state(loss, sample_weight=sample_weight)
        self.valid_metric.update_state(outputs, predictions)

    @tf.function
    def distributed_valid_step(self, model, inputs, outputs, active_entries):
        self.strategy.run(self.valid_step, args=(model, inputs, outputs, active_entries))

    def train_model(self, model, params, 
                    use_truncated_bptt=True, 
                    b_stub_front=True,
                    b_use_state_initialisation=True):

        # Create a checkpoint
        #checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        # initialize history
        history = {
            'train_loss': [],
            'train_mse': [],
            'valid_loss': [],
            'valid_mse': []
            }
        min_epochs = 50
        min_loss = tf.constant(np.inf)
        for epoch in tf.range(1, self.epochs+1):
            total_loss = 0.0
            num_batches = 0
  
            for data in self.ds_train:
                
                input_data = data['inputs']
                output_data = data['outputs']
                active_entries = data['active_entries']
                weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)

                # Stack up the dynamic RNNs for T-BPTT.
                # Splitting it up
                total_timesteps = input_data.get_shape().as_list()[1]
                num_slices = int(total_timesteps / self.backprop_length)
                chunk_sizes = [self.backprop_length for i in range(num_slices)]
                odd_size = total_timesteps - self.backprop_length*num_slices

                # get all the chunks
                if odd_size > 0:
                    if b_stub_front:
                        chunk_sizes = [odd_size] + chunk_sizes
                    else:
                        chunk_sizes = chunk_sizes + [odd_size]

                # Inplement TF style Truncated-backprop through time
                #self.train_step(model, input_data, output_data, active_entries, weights, chunk_sizes)
                loss, predictions, outputs = self.distributed_train_step(model, input_data, output_data, active_entries, weights, chunk_sizes)
                #print("predictions:")
                #print(predictions)
                #print("outputs:")
                #print(outputs)
                total_loss += loss
                num_batches += 1
          
            train_loss = total_loss / num_batches

            for data in self.ds_valid:
                #self.valid_step(model, data['inputs'], data['outputs'], data['active_entries'])
                self.distributed_valid_step(model, data['inputs'], data['outputs'], data['active_entries'])

            #if tf.math.is_nan(self.valid_loss.result()):
            #    logging.warning("NAN Loss found, terminating routine")
            #    break

            # save history
            history['train_loss'].append(train_loss) # train_loss
            history['train_mse'].append(self.train_metric.result().numpy())
            history['valid_loss'].append(self.valid_loss.result().numpy())
            history['valid_mse'].append(self.valid_metric.result().numpy())


            # save optimal results
            if self.valid_loss.result() < min_loss and epoch > min_epochs:
                min_loss = self.valid_loss.result()
                save_model(model, params, history, option='optimal')

            # looging and state reset
            logs = 'Epoch={}/{},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{} | {}'
        
            if epoch%1 ==0:
                printbar()
                tf.print(tf.strings.format(logs,
                (epoch, self.epochs, train_loss, self.train_metric.result(),self.valid_loss.result(),self.valid_metric.result(), params['net_name']))) # train_loss
                tf.print("")

            #if epoch%2 == 0:
            #    checkpoint.save(os.path.join(self.checkpoint_dir, "ckpt"))
            
            #self.train_loss.reset_states()
            self.valid_loss.reset_states()
            self.train_metric.reset_states()
            self.valid_metric.reset_states()

        return history

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


    #def evaluate_model(self, model):
    #    total_loss = 0
    #    total_metric = 0
    #    num_batches = 0

    #    for data in self.ds_test:
    #        self.distributed_valid_step(model, data['inputs'], data['outputs'], data['active_entries'])
    #        total_loss += self.valid_loss.result().numpy()
    #        total_metric += self.valid_metric.result().numpy()
    #        num_batches += 1

    #        self.valid_loss.reset_states()
    #        self.valid_metric.reset_states()

    #    # Calculate the average loss and metrics for the entire dataset
    #    avg_loss = total_loss / num_batches
    #    avg_metric = total_metric / num_batches
    #    return avg_loss, avg_metric # avg_metric=mse

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

def model_predict(model, dataset, mc_sampling=False):
    # Initialize lists to store final statistics for all chunks
    all_means = []
    all_upper_bounds = []
    all_lower_bounds = []
    logs = 'Predicting ' + model.name
    if mc_sampling:
        pred_times = 100
    else:
        pred_times = 1

    for data_chunk in tqdm(dataset, desc=logs):
        chunk_predictions = []
        batch_size = tf.shape(data_chunk['inputs'])[0]
        hidden_layer_size = model.get_layer('lstm').units
        initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
        # Predict the current chunk multiple times
        for _ in range(pred_times):
            prediction, _, _ = model.predict([data_chunk['inputs'], initial_state, initial_state], verbose=0)
            chunk_predictions.append(prediction)

        # Convert list of predictions to a numpy array for statistical computation
        chunk_predictions = np.array(chunk_predictions)

        # Calculate mean, upper bound, and lower bound for the current chunk
        mean_estimate = np.mean(chunk_predictions, axis=0)
        upper_bound = np.percentile(chunk_predictions, 95, axis=0)
        lower_bound = np.percentile(chunk_predictions, 5, axis=0)

        # Append the statistics of the current chunk to their respective lists
        all_means.append(mean_estimate)
        all_upper_bounds.append(upper_bound)
        all_lower_bounds.append(lower_bound)

    # Optional: Convert lists to numpy arrays if further processing is needed
    all_means = np.concatenate(all_means, axis=0) if all_means else np.array([])
    all_upper_bounds = np.concatenate(all_upper_bounds, axis=0) if all_upper_bounds else np.array([])
    all_lower_bounds = np.concatenate(all_lower_bounds, axis=0) if all_lower_bounds else np.array([])

    # At this point, you can either return the raw statistics for each chunk,
    # or aggregate them in some way depending on your application's needs.
    # The following returns the list of statistics for all chunks.
    return {
        'mean_pred': all_means,
        'upper_bound': all_upper_bounds,
        'lower_bound': all_lower_bounds
    }

# Step 5: Save and Load Model
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

def save_model(model, params, history, option='optimal'): # option: final or optimal
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
    serialisation_name = serialisation_name + "_" + option
    model_path = os.path.join(model_folder, serialisation_name)
    model.save(model_path, save_format = 'tf')
    
    # save history (if optimal)
    if option == 'optimal':
        history = pd.DataFrame(history)
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
