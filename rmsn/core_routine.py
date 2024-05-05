import numpy as np
import pandas as pd
import os
import logging
import math
from six import b
from tensorflow.python.keras.distribute.distributed_training_utils import global_batch_size_supported
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import *
from rmsn.libs.data_process import convert_to_tf_dataset, convert_to_tf_dataset_via_generator
from rmsn.libs.rmsn_model import create_model, CustomLoss
from rmsn.configs import strategy
# Set up mirrored strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

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

# Truncated-backprop through time
def truncated_BPTT(model, inputs, input_size, chunk_sizes, hidden_layer_size):
    segment_predictions = []
    start = 0
    states = None
    for i, chunk_size in enumerate(chunk_sizes):
        input_chunk = tf.slice(inputs, [0, start, 0], [-1, chunk_size, input_size])
        if states is None:
            batch_size = tf.shape(input_chunk)[0]
            initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
            segment_prediction, state_h, state_c = model([input_chunk, initial_state, initial_state], training=True)
        else:
            segment_prediction, state_h, state_c = model([input_chunk, states[0], states[1]], training=True)
           
        segment_predictions.append(segment_prediction)
        # break links between states for truncated bptt
        states = [state_h, state_c]
        #states = tf.identity(states)
        # Starting point
        start += chunk_size

    # Dumping output
    predictions = tf.concat(segment_predictions, axis=1)
    return predictions

def propensity_model_train(params):
    # data pipeline
    training_processed = params['training_dataset']
    validation_processed = params['validation_dataset']
    global_batch_size = params['minibatch_size'] * strategy.num_replicas_in_sync
    # data generator
    tf_data_train = convert_to_tf_dataset_via_generator(training_processed, global_batch_size)
    tf_data_valid = convert_to_tf_dataset_via_generator(validation_processed, global_batch_size)
    # distribute them
    tf_data_train = strategy.experimental_distribute_dataset(tf_data_train)
    tf_data_valid = strategy.experimental_distribute_dataset(tf_data_valid)

    # Create a checkpint directory to store the checkpoints
    model_folder = params['model_folder']
    checkpoint_prefix = os.path.join(model_folder, "ckpt")

    # initialize model and loss func
    with strategy.scope():
        model = create_model(params)
        model.summary()
        #optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
        optimizer = optimizers.SGD(learning_rate=params['learning_rate'])
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        multiclass_loss_params = {'performance_metric': 'sparse_xentropy',
                                  'num_gpus': strategy.num_replicas_in_sync,
                                  'global_batch_size': global_batch_size,
                                 }
        multiclass_loss_func = CustomLoss(multiclass_loss_params)

        binary_loss_params = {'performance_metric': 'xentropy',
                              'num_gpus': strategy.num_replicas_in_sync,
                              'global_batch_size': global_batch_size
                              }
        binary_loss_func = CustomLoss(binary_loss_params)

        multiclass_train_metric = metrics.SparseCategoricalAccuracy(name='multiclass_train_metric')
        binary_train_metric = metrics.BinaryAccuracy(name='binary_train_metric')

        valid_loss = metrics.Mean(name='valid_loss')
        multiclass_valid_metric = metrics.SparseCategoricalAccuracy(name='multiclass_valid_metric')
        binary_valid_metric = metrics.BinaryAccuracy(name='binary_valid_metric')

    # train and valid functions ###################################################################################
    def train_step(data, chunk_sizes):
        inputs = data['inputs']
        outputs = data['outputs']
        multiclass_outputs, binary_outputs = tf.split(outputs, [num_continuous, output_size - num_continuous], axis=-1)
        active_entries = data['active_entries']
        weights = data['propensity_weights'] if 'propensity_weights' in data else None

        with tf.GradientTape() as tape:
            # T-BPTT
            predictions = truncated_BPTT(model, inputs, input_size, chunk_sizes, hidden_layer_size)
            multiclass_preds, binary_preds = tf.split(predictions, [softmax_size, predict_size - softmax_size], axis=-1)
            active_entries1, active_entries2 = tf.split(active_entries, [num_continuous, output_size - num_continuous], axis=-1)

            multiclass_loss = multiclass_loss_func.train_call(multiclass_outputs, multiclass_preds, active_entries1, weights)
            # 加入判断，处理没有policy的情况
            binary_loss = binary_loss_func.train_call(binary_outputs, binary_preds, active_entries2, weights) if num_continuous < output_size else 0
            total_loss = multiclass_loss + binary_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = max_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # self.train_loss.update_state(total_loss)
        # self.continuous_train_loss.update_state(continuous_loss)
        # self.binary_train_loss.update_state(binary_loss)
        multiclass_train_metric.update_state(multiclass_outputs, multiclass_preds)
        binary_train_metric.update_state(binary_outputs, binary_preds) # need modified

        return total_loss

    @tf.function
    def distributed_train_step(data, chunk_sizes):
        per_replica_losses = strategy.run(train_step, args=(data, chunk_sizes)) 
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def valid_step(data):
        
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']

        multiclass_outputs, binary_outputs = tf.split(outputs, [num_continuous, output_size - num_continuous], axis=-1) 
        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
        predictions,_,_ = model([inputs,initial_state, initial_state], training=False)
        multiclass_preds, binary_preds = tf.split(predictions, [softmax_size, predict_size - softmax_size], axis=-1)

        multiclass_loss = multiclass_loss_func.valid_call(multiclass_outputs, multiclass_preds)
        binary_loss = binary_loss_func.valid_call(binary_outputs, binary_preds) if num_continuous < output_size else 0
        total_loss = multiclass_loss + binary_loss
        
        valid_loss.update_state(total_loss)
        multiclass_valid_metric.update_state(multiclass_outputs, multiclass_preds)
        binary_valid_metric.update_state(binary_outputs, binary_preds)

    @tf.function
    def distributed_valid_step(data):
        strategy.run(valid_step, args=(data,))
    # ###########################################################################################

    # Training Loop
    hidden_layer_size = params['hidden_layer_size']
    max_norm = params['max_norm']
    input_size = params['input_size']
    output_size = params['output_size']
    num_continuous = params['num_continuous']
    softmax_size = params['softmax_size']
    predict_size = params['predict_size']
    epochs = params['num_epochs']
    backprop_length = params['backprop_length']
    total_timesteps = params['time_steps']
    history = {
        'train_loss': [],
        'multiclass_train_metric': [],
        'binary_train_metric':[],
        'valid_loss':[],
        'multiclass_valid_metric':[],
        'binary_valid_metric':[]
        }
    min_epochs = 70
    min_loss = tf.constant(np.inf)
    b_stub_front = False

    for epoch in tf.range(1, epochs+1):
        total_loss = 0
        num_batches = 0
  
        for data in tf_data_train:
                
            input_data = data['inputs']
            #output_data = data['outputs']
            #active_entries = data['active_entries']
            #weights = data['propensity_weights'] if 'propensity_weights' in data else tf.constant(1.0)

            # Stack up the dynamic RNNs for T-BPTT.
            # Splitting it up
            #print(data['sequence_lengths'])
            num_slices = int(total_timesteps / backprop_length)
            chunk_sizes = [backprop_length for i in range(num_slices)]
            odd_size = total_timesteps - backprop_length * num_slices

            # get all the chunks
            if odd_size > 0:
                if b_stub_front:
                    chunk_sizes = [odd_size] + chunk_sizes
                else:
                    chunk_sizes = chunk_sizes + [odd_size]

            # Inplement TF style Truncated-backprop through time
            #self.train_step(model, input_data, output_data, active_entries, weights, chunk_sizes)
            total_loss += distributed_train_step(data, chunk_sizes)
            num_batches += 1

        train_loss = total_loss / num_batches

        for data in tf_data_valid:
            distributed_valid_step(data)

        if tf.math.is_nan(valid_loss.result()):
            logging.warning("NAN Loss found, terminating routine")
            break

        # save history
        history['train_loss'].append(train_loss) # train_loss
        history['multiclass_train_metric'].append(multiclass_train_metric.result().numpy())
        history['binary_train_metric'].append(binary_train_metric.result().numpy())
        history['valid_loss'].append(valid_loss.result().numpy())
        history['multiclass_valid_metric'].append(multiclass_valid_metric.result().numpy())
        history['binary_valid_metric'].append(binary_valid_metric.result().numpy())


        # save optimal results
        total_valid_loss  = valid_loss.result()
        if total_valid_loss < min_loss and epoch > min_epochs:
            min_loss = total_valid_loss
            save_model(model, params, history, option='optimal')

        # looging and state reset
        logs = 'Epoch={}/{},Loss:{},Metric:[{},{}],Valid_Loss:{},Valid_Metric:[{},{}] | {}'
        
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch, epochs, train_loss, multiclass_train_metric.result(), binary_train_metric.result(),
             valid_loss.result(), multiclass_valid_metric.result(), binary_valid_metric.result(), params['net_name']))) # train_loss
            tf.print("")

        if epoch%2 == 0:
            checkpoint.save(checkpoint_prefix)
            
        #self.train_loss.reset_states()
        valid_loss.reset_states()
        multiclass_train_metric.reset_states()
        binary_train_metric.reset_states()
        multiclass_valid_metric.reset_states()
        binary_valid_metric.reset_states()

    # Save the final model
    save_model(model, params, history, option='final')

    return history

def predictive_model_train(params):
    # data pipeline
    training_processed = params['training_dataset']
    validation_processed = params['validation_dataset']
    global_batch_size = params['minibatch_size'] * strategy.num_replicas_in_sync
    performance_metric = params['performance_metric']

    # data generator
    tf_data_train = convert_to_tf_dataset_via_generator(training_processed, global_batch_size)
    tf_data_valid = convert_to_tf_dataset_via_generator(validation_processed, global_batch_size)
    # distribute them
    tf_data_train = strategy.experimental_distribute_dataset(tf_data_train)
    tf_data_valid = strategy.experimental_distribute_dataset(tf_data_valid)

    # Create a checkpint directory to store the checkpoints
    model_folder = params['model_folder']
    checkpoint_prefix = os.path.join(model_folder, "ckpt")
    # initialize model and loss func
    with strategy.scope():
        model = create_model(params)
        model.summary()
        #optimizer = optimizers.Adam(learning_rate=params['learning_rate'])
        optimizer = optimizers.SGD(learning_rate=params['learning_rate'])
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        loss_params = {'performance_metric': performance_metric,
                       'num_gpus': strategy.num_replicas_in_sync,
                       'global_batch_size': global_batch_size,
                       }
        loss_func = CustomLoss(loss_params)
        train_metric = metrics.MeanSquaredError(name='train_mse')

        valid_loss = metrics.Mean(name='valid_loss')
        valid_metric = metrics.MeanSquaredError(name='valid_mse')

    # train and valid functions ###################################################################################
    def train_step(data, chunk_sizes):
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']
        weights = data['propensity_weights'] if 'propensity_weights' in data else None

        with tf.GradientTape() as tape:
            # T-BPTT
            predictions = truncated_BPTT(model, inputs, input_size, chunk_sizes, hidden_layer_size)
            loss = loss_func.train_call(outputs, predictions, active_entries, weights)

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm = max_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # self.train_loss.update_state(total_loss)
        # self.continuous_train_loss.update_state(continuous_loss)
        # self.binary_train_loss.update_state(binary_loss)
        train_metric.update_state(outputs, predictions)

        return loss

    @tf.function
    def distributed_train_step(data, chunk_sizes):
        per_replica_losses = strategy.run(train_step, args=(data, chunk_sizes)) 
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    #@tf.function
    def valid_step(data):
        
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']

        batch_size = tf.shape(inputs)[0]
        initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
        predictions, _, _ = model([inputs, initial_state, initial_state], training=False)

        loss = loss_func.valid_call(outputs, predictions)
        sample_weight = active_entries/tf.reduce_sum(active_entries)
    
        valid_loss.update_state(loss, sample_weight=sample_weight)
        valid_metric.update_state(outputs, predictions)

    @tf.function
    def distributed_valid_step(data):
        strategy.run(valid_step, args=(data,))
    # ###########################################################################################

    # Training Loop
    hidden_layer_size = params['hidden_layer_size']
    max_norm = params['max_norm']
    input_size = params['input_size']
    output_size = params['output_size']
    total_timesteps = params['time_steps']
    epochs = params['num_epochs']
    backprop_length = params['backprop_length']
    history = {
        'train_loss': [],
        'train_mse': [],
        'valid_loss':[],
        'valid_mse': []
        }
    min_epochs = 50
    min_loss = tf.constant(np.inf)
    b_stub_front = False

    for epoch in tf.range(1, epochs+1):
        total_loss = 0
        num_batches = 0
  
        for data in tf_data_train:
            input_data = data['inputs']

            # Stack up the dynamic RNNs for T-BPTT.
            # Splitting it up
            #print(data['sequence_lengths'])
            num_slices = int(total_timesteps / backprop_length)
            chunk_sizes = [backprop_length for i in range(num_slices)]
            odd_size = total_timesteps - backprop_length*num_slices

            ## get all the chunks
            if odd_size > 0:
                if b_stub_front:
                    chunk_sizes = [odd_size] + chunk_sizes
                else:
                    chunk_sizes = chunk_sizes + [odd_size]

            # Inplement TF style Truncated-backprop through time
            #self.train_step(model, input_data, output_data, active_entries, weights, chunk_sizes)
            total_loss += distributed_train_step(data, chunk_sizes)
            num_batches += 1
          
        train_loss = total_loss / num_batches

        for data in tf_data_valid:
            distributed_valid_step(data)

        #if tf.math.is_nan(self.valid_loss.result()):
        #    logging.warning("NAN Loss found, terminating routine")
        #    break

        # save history
        history['train_loss'].append(train_loss) # train_loss
        history['train_mse'].append(train_metric.result().numpy())
        history['valid_loss'].append(valid_loss.result().numpy())
        history['valid_mse'].append(valid_metric.result().numpy())

        if valid_loss.result().numpy() < min_loss and epoch > min_epochs:
            min_loss = valid_loss.result().numpy()
            save_model(model, params, history, option='optimal')

        # looging and state reset
        logs = 'Epoch={}/{},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{} | {}'
        
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch, epochs, train_loss, train_metric.result(),valid_loss.result(),valid_metric.result(), params['net_name']))) # train_loss
            tf.print("")

        if epoch%2 == 0:
            checkpoint.save(checkpoint_prefix)
            
        #self.train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

    # Save the final model
    save_model(model, params, history, option='final')

    return history

def model_predict(model, dataset, hidden_layer_size, mc_sampling=False):
    # Initialize lists to store final statistics for all chunks
    Predictions = []
    logs = 'Predicting ' + model.name
    if mc_sampling:
        pred_times = 10
    else:
        pred_times = 1

    for _ in range(pred_times):
        predictions = []
        for data_chunk in tqdm(dataset, desc=logs):
            batch_size = tf.shape(data_chunk['inputs'])[0]
            initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
            # Predict the current chunk multiple times
            chunk_prediction, _, _ = model.predict_on_batch([data_chunk['inputs'], initial_state, initial_state])
            # print(prediction.shape)
                # chunk_predictions.append(prediction)

            # Convert list of predictions to a numpy array for statistical computation
            # chunk_predictions = np.array(chunk_predictions)
            predictions.extend(chunk_prediction)
        
        if mc_sampling:
            Predictions.append(predictions)
    
    if mc_sampling:
        return np.stack(Predictions,axis=0)
    else:
        return np.array(predictions)


def propensity_predict(params):
    
    model_folder = params['model_folder']
    serialisation_name = params['serialisation_name']
    training_dataset = params['training_dataset']
    batch_size = params['minibatch_size']
    num_continuous = params['num_continuous']
    softmax_size = params['softmax_size']
    hidden_layer_size = params['hidden_layer_size']  # use for initializing states

    # load model
    model = load_model(model_folder, serialisation_name)

    # data pipeline
    #global_batch_size = batch_size * strategy.num_replicas_in_sync
    tf_data = convert_to_tf_dataset_via_generator(training_dataset, batch_size, for_train=False)

     # predictition
    outputs = training_dataset['scaled_outputs']
    multiclass_outputs, binary_outputs = np.split(outputs, [num_continuous], axis=-1)
    
    # 对binary变量预测一次就行，对continuous变量则应用mc dropout
    # for binary
    results = model_predict(model, tf_data, hidden_layer_size)
    multiclass_preds, binary_preds = np.split(results, [softmax_size], axis=-1)
    
    ## for continuous
    #mc_results_mean, mc_results_std = model_predict(model, tf_data, hidden_layer_size, mc_sampling=True)
    #continuous_mean, _ = np.split(mc_results_mean, [num_continuous], axis=-1)
    #continuous_std, _ = np.split(mc_results_std, [num_continuous], axis=-1)

    return multiclass_preds, multiclass_outputs, binary_preds, binary_outputs

def effect_predict(params):
    
    model_folder = params['model_folder']
    serialisation_name = params['serialisation_name']
    dataset = params['dataset']
    batch_size = params['minibatch_size']
    hidden_layer_size = params['hidden_layer_size']  # use for initializing states

    # load model
    model = load_model(model_folder, serialisation_name)

    # data pipeline
    #global_batch_size = batch_size * strategy.num_replicas_in_sync
    tf_data = convert_to_tf_dataset_via_generator(dataset, batch_size, for_train=False)

     # predictition
    results = model_predict(model, tf_data, hidden_layer_size)

    return results

def model_evaluate(params):

    eval_accuracy = metrics.RootMeanSquaredError(name='eval_rmse')

    new_model = create_model(params)
    new_optimizer = optimizers.Adam()

    test_processed = params['test_dataset']
    global_batch_size = params['minibatch_size'] * strategy.num_replicas_in_sync

    # 需要修改为data generator
    tf_data_test = convert_to_tf_dataset(test_processed, global_batch_size)

    @tf.function
    def eval_step(data):
        inputs = data['inputs']
        outputs = data['outputs']
        active_entries = data['active_entries']

        batch_size = tf.shape(inputs)[0]
        hidden_layer_size = params['hidden_layer_size']
        initial_state = tf.zeros([batch_size, hidden_layer_size], dtype=tf.float32)
        predictions, _, _ = new_model([inputs, initial_state, initial_state], training=False)

        sample_weight = active_entries/tf.reduce_sum(active_entries)
        eval_accuracy.update_state(outputs, predictions, sample_weight=sample_weight)
        
    
    checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
    checkpoint_dir = params['model_folder']
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for data in tf_data_test:
        eval_step(data)
    
    return eval_accuracy.result()*100

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
    #with strategy.scope():
    model = models.load_model(model_path, compile = False)

    logging.info("Successfully loaded model from {}".format(model_path))

    return model