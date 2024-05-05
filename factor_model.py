import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from utils.predictive_checks_utils import compute_test_statistic_all_timesteps
from utils.rnn_utils import *
from rmsn.configs import strategy

# mirrored strategy
strategy = tf.distribute.MirroredStrategy()
logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))


class FactorModel:
    def __init__(self, params, hyperparams):
        
        self.num_treatments = params['num_treatments']
        self.num_covariates = params['num_covariates']
        self.num_confounders = params['num_confounders']
        self.max_sequence_length = params['max_sequence_length']
        self.num_epochs = params['num_epochs']

        self.rnn_hidden_units = hyperparams['rnn_hidden_units']
        self.fc_hidden_units = hyperparams['fc_hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.rnn_keep_prob = hyperparams['rnn_keep_prob']

    def build_confounders(self, previous_covariates_input, previous_treatments_input, current_covariates_input, trainable_init_input):
        # 合并输入
        previous_covariates_and_treatments = layers.Concatenate(axis=-1)([previous_covariates_input, previous_treatments_input])
        rnn_input = layers.Concatenate(axis=1, name="rnn_input")([trainable_init_input, previous_covariates_and_treatments])

        # LSTM网络
        # 增加dropout机制
        lstm_cell = DropoutLSTMCell(self.rnn_hidden_units, dropout=1-self.rnn_keep_prob, recurrent_dropout=1-self.rnn_keep_prob)
        autoregressice_cell = AutoregressiveLSTMCell(lstm_cell, self.num_confounders)
        rnn_layer = layers.RNN(autoregressice_cell, return_sequences=True, name="hidden_confounders")
        x = rnn_layer(rnn_input)

        # 处理当前协变量和隐藏混杂因素
        hidden_confounders = x
        covariates = current_covariates_input
        # hidden_confounders = layers.Reshape(target_shape=(-1, num_confounders))(x)
        # covariates = layers.Reshape(target_shape=(-1, num_covariates))(current_covariates_input)
        multitask_input = layers.Concatenate(axis=-1)([covariates, hidden_confounders])
    
        return multitask_input, rnn_input, hidden_confounders

    def build_treatment_assignments(self, multitask_input):
    
        shared_fc_layer = layers.Dense(self.fc_hidden_units, activation=tf.nn.leaky_relu, name='shared_fc_layer')(multitask_input)
    
        binary_treatment_outputs = []
        continuous_treatment_outputs = []
    
        # 二元处置
        for _ in range(self.num_binary_treatments):
            binary_treatment_output = layers.Dense(1, activation=tf.nn.sigmoid)(shared_fc_layer)
            binary_treatment_outputs.append(binary_treatment_output)
        # 合并治疗预测
        binary_treatment_prob_predictions = layers.Concatenate(axis=-1, name='binary_treatment_prob_predictions')\
                                            (binary_treatment_outputs)
    
        # 连续处置，MC dropout
        for _ in range(self.num_continuous_treatments):
            continuous_treatment_output = layers.Dense(1)(shared_fc_layer)
            continuous_treatment_output = layers.Dropout(1-self.rnn_keep_prob)(continuous_treatment_output, training=True)
            continuous_treatment_outputs.append(continuous_treatment_output)
        continuous_treatment_prob_predictions = layers.Concatenate(axis=-1, name='continuous_treatment_prob_predictions')\
                                            (continuous_treatment_outputs)
    
        return binary_treatment_prob_predictions, continuous_treatment_prob_predictions

    def build_network(self):  
    
        # 定义输入
        previous_covariates_input = layers.Input(shape=(None, self.num_covariates), dtype=tf.float32, name="previous_covariates")
        previous_treatments_input = layers.Input(shape=(None, self.num_treatments), dtype=tf.float32, name="previous_treatments")
        current_covariates_input = layers.Input(shape=(self.max_sequence_length, self.num_covariates), dtype=tf.float32, name="covariates")
        # trainable_init_input = tf.Variable(tf.random.normal([batch_size, 1, num_covariates + num_treatments]), trainable=True)
        trainable_init_input_layer = TrainableInitialInput(output_dim=self.num_covariates + self.num_treatments)
        trainable_init_input = trainable_init_input_layer(previous_covariates_input)

        # 构建confounders
        multitask_input, rnn_input, hidden_confounders = self.build_confounders(previous_covariates_input, previous_treatments_input, 
                                                                                current_covariates_input, trainable_init_input)

        # 构建treatment assignments
        binary_treatment_prob_predictions, continuous_treatment_prob_predicitons = \
                            self.build_treatment_assignments(multitask_input)      
    
        binary_combined_outputs = layers.Concatenate(axis=-1, name="binary_combined_output")\
                                    ([rnn_input, binary_treatment_prob_predictions])
        continuous_combined_outputs = layers.Concatenate(axis=-1, name="continuous_combined_output")\
                                    ([rnn_input, continuous_treatment_prob_predicitons])

        # 定义模型
        model = tf.keras.Model(inputs=[previous_covariates_input, previous_treatments_input, current_covariates_input],
                               outputs=[binary_combined_outputs, continuous_combined_outputs, hidden_confounders],
                               name="factor_model")
    
        return model

    def compute_test_statistic(self, dataset, num_samples, target_treatments, predicted_mask):
        
        test_statistic = np.zeros(shape=(self.max_sequence_length,))
        num_inputs = self.num_treatments + self.num_covariates

        for sample_idx in range(num_samples):
            combined_outputs, _ = self.model.predict(dataset)
            treatment_probability = combined_outputs[...,num_inputs:]

            test_statistic_sequence = compute_test_statistic_all_timesteps(target_treatments,
                                                                           treatment_probability,
                                                                           self.max_sequence_length, predicted_mask)
            test_statistic += test_statistic_sequence

        test_statistic = test_statistic / num_samples

        return test_statistic

    # 还没有实现对连续值处置的处理
    def eval_predictive_checks(self, dataset):
        num_replications = 50
        num_samples = 50
        target_treatments = dataset['treatments']
        tf_data = convert_to_tf_dataset(dataset, self.batch_size)

        # model predict
        outputs = self.model.predict(dataset)
        rnn_input, binary_treatment_prob_predictions, continuous_treatment_prob_predictions, _ = self.split_output(outputs)
        mask = np.sign(np.max(np.abs(rnn_input), axis=2))
    
        test_statistic_replicas = np.zeros(shape=(num_replications, max_sequence_length))
        for replication_idx in range(num_replications):
            # 这里我们生成伯努利实现，假设 treatment_prob_pred 已经是适当的形状
            treatment_replica = np.random.binomial(n=1, p=binary_treatment_prob_predictions)
            # 计算测试统计量
            test_statistic_replicas[replication_idx] = compute_test_statistic(dataset, num_samples, treatment_replica, mask)
    
        test_statistic_target = compute_test_statistic(dataset, num_samples, target_treatments, mask)
    
        p_values_over_time = np.mean(np.less(test_statistic_replicas, test_statistic_target).astype(np.int32), axis=0)
        return p_values_over_time

    def train(self, dataset_train, dataset_val, verbose=2):

        # data preparation
        # compute numbers of binary and continuous treatments (default: continuous variables first)
        self.num_binary_treatments = 0
        self.num_continuous_treatments = 0
        for index in range(self.num_treatments):
            treatment = dataset_train['treatments'][:,:,index]
            if np.all(np.isin(treatment, [0, 1])):
                self.num_binary_treatments += 1
            else:
                self.num_continuous_treatments += 1

        self.global_batch_size = self.batch_size*strategy.num_replicas_in_sync
        #tf_train_data = convert_to_tf_dataset(dataset_train, global_batch_size)
        #tf_val_data = convert_to_tf_dataset(dataset_val, global_batch_size)
        # use data generator
        train_gen = DataGenerator(dataset_train, self.global_batch_size)
        val_gen = DataGenerator(dataset_val, self.global_batch_size)
        

        with strategy.scope():
            # build model
            self.model = self.build_network()
            self.model.summary()
            
            self.model.compile(optimizer=self.get_optimizer(), 
              loss={'binary_combined_output':BinaryLoss(self.num_covariates + self.num_treatments, self.num_continuous_treatments),
                    'continuous_combined_output':ContinuousLoss(self.num_covariates + self.num_treatments, self.num_continuous_treatments),
                   'hidden_confounders':None}, 
              metrics={'binary_combined_output':'accuracy',
                      'continuous_combined_output':'accuracy'}) 

        # begin training
        # set up early stop callback
        #early_stopping_callback = EarlyStopping(monitor='val_loss',
        #                                patience=10,
        #                                verbose=1, 
        #                                mode='min', 
        #                                restore_best_weights=True)
        # set up nan callback
        nan_terminate_callback = TerminateOnNaN()
        callbacks = nan_terminate_callback
        # Train model
        start_time = time.time()
        history = self.model.fit(train_gen, 
                            epochs=self.num_epochs, 
                            verbose=verbose, 
                            validation_data=val_gen,
                            callbacks=callbacks)

        end_time = time.time()
        training_time = end_time - start_time
        logging.info("Training time: {:.2f} seconds".format(training_time))

        ## Save the trained model
        #self.model.save('results/factor_model', save_format="tf")
        #logging.info('export saved model.')

    def get_optimizer(self):
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        return optimizer

    def split_output(self, predictions):
        binary_combined_outputs, continuous_combined_outputs, hidden_confounders = predictions
        num_input = self.num_treatments + self.num_covariates
        binary_treatment_prob_predictions = binary_combined_outputs[...,num_input:]
        rnn_input = binary_treatment_prob_predictions[...,:num_input]
        continuous_treatment_prob_predictions = continuous_combined_outputs[...,num_input:]
        return rnn_input, binary_treatment_prob_predictions, continuous_treatment_prob_predictions, hidden_confounders
    
    def compute_hidden_confounders(self, dataset):
        # load model
        #if not hasattr(self, 'model'):
        #    self.model = models.load_model('results/factor_model', compile=False)
        #    self.model.compile(optimizer=self.get_optimizer(), 
        #          loss={'binary_combined_output':BinaryLoss(self.num_covariates + self.num_treatments, self.num_continuous_treatments),
        #                'continuous_combined_output':ContinuousLoss(self.num_covariates + self.num_treatments, self.num_continuous_treatments),
        #               'hidden_confounders':None}, 
        #          metrics={'binary_combined_output':'accuracy',
        #                  'continuous_combined_output':'accuracy'}) 

        #    #tf_data = convert_to_tf_dataset(dataset, self.batch_size)
        #    self.global_batch_size = self.batch_size*strategy.num_replicas_in_sync
        
        gen = DataGenerator(dataset, self.global_batch_size)
        hidden_confounders = []
        for i in range(len(gen)):
            features, _ = gen[i]
            _, _, batch_confounder = self.model.predict(features)
            hidden_confounders.extend(batch_confounder)
        hidden_confounders = np.array(hidden_confounders)
        #_, _, _, hidden_confounders = self.split_output(predictions)

        return hidden_confounders