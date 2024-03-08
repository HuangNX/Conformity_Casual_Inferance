import tensorflow as tf
from tensorflow.keras import *

# data module #################################################
def map_function(features):
    # 提取输入
    inputs = {
        "previous_covariates": features['previous_covariates'],
        "previous_treatments": features['previous_treatments'],
        "covariates": features['covariates']
    }
    # 提取输出
    outputs = features['treatments']
    
    return inputs, outputs

def convert_to_tf_dataset(dataset, batch_size):
    key_map = {'previous_covariates': dataset['previous_covariates'],
               'previous_treatments': dataset['previous_treatments'],
               'covariates': dataset['covariates'],
               'treatments': dataset['treatments'],
               'sequence_length': dataset['sequence_length']}

    #from_tensor_slices:切片; shuffle:随机打乱; batch:批次组合; prefetch:提前准备（预取）数据
    tf_dataset = tf.data.Dataset.from_tensor_slices(key_map)\
                .map(map_function)\
                .batch(batch_size) \
                .prefetch(tf.data.experimental.AUTOTUNE)
    #.shuffle(buffer_size = 1000)\

    return tf_dataset

# Loss function module ##########################################
# defined loss function
class BinaryLoss(tf.keras.losses.Loss):
    def __init__(self, num_input, num_continuous_treatment, name="binary_loss"):
        super(BinaryLoss, self).__init__(name=name)
        self.num_input = num_input
        self.num_continuous_treatment = num_continuous_treatment

    def call(self, y_true, y_pred):
        y_true = y_true[...,self.num_continuous_treatment:]
        
        rnn_input = y_pred[...,:self.num_input]
        y_pred = y_pred[...,self.num_input:]
        
        # 计算掩码，忽略padding的影响
        mask = tf.sign(tf.reduce_max(tf.abs(rnn_input), axis=-1, keepdims=True))
        flat_mask = tf.reshape(mask, [-1, 1])
        
        # 计算交叉熵损失
        cross_entropy = -tf.reduce_sum(
            (y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) + 
                (1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-10, 1.0))) 
            * mask, axis=1)
        
        # 根据序列长度对损失进行归一化
        sequence_length = compute_sequence_length(rnn_input)
        cross_entropy /= tf.reduce_sum(tf.cast(sequence_length, tf.float32), axis=0)
        # tf.print("finish loss computation.")
        return tf.reduce_mean(cross_entropy)

class ContinuousLoss(tf.keras.losses.Loss):
    def __init__(self, num_input, num_continuous_treatment, name="continuous_loss"):
        super(ContinuousLoss, self).__init__(name=name)
        self.num_input = num_input
        self.num_continuous_treatment = num_continuous_treatment

    def call(self, y_true, y_pred):
        y_true = y_true[...,:self.num_continuous_treatment]
        
        # 假设 y_pred 包含输入和预测值
        rnn_input = y_pred[...,:self.num_input]
        y_pred = y_pred[...,self.num_input:]
        
        # 计算掩码，忽略 padding 的影响
        mask = tf.sign(tf.reduce_max(tf.abs(rnn_input), axis=-1, keepdims=True))
        flat_mask = tf.reshape(mask, [-1, 1])
        
        # 计算连续变量的 RMSE 损失
        squared_difference = tf.square(y_true - y_pred) * mask  # 应用掩码
        mse_loss = tf.reduce_sum(squared_difference, axis=1) / tf.reduce_sum(flat_mask, axis=1)  # 对损失进行归一化
        
        rmse_loss = tf.sqrt(mse_loss)  # 计算 RMSE
        
        return tf.reduce_mean(rmse_loss)


# DIY Layer module #############################################
# drop out layer
class DropoutLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, dropout=0., recurrent_dropout=0., **kwargs):
        super(DropoutLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # 适用于输入的 dropout 层
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        # 适用于循环连接的 dropout 层
        self.recurrent_dropout_layer = tf.keras.layers.Dropout(recurrent_dropout)
        self.state_size = self.lstm_cell.state_size
        self.output_size = self.lstm_cell.output_size

    def call(self, inputs, states, training=None):
        if training:
            inputs = self.dropout_layer(inputs, training=training)
            states = [self.recurrent_dropout_layer(state, training=training) for state in states]
        return self.lstm_cell(inputs, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.lstm_cell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)

# auto regressive layer
class AutoregressiveLSTMCell(tf.keras.layers.Layer):
    def __init__(self, lstm_cell, output_size, **kwargs):
        super(AutoregressiveLSTMCell, self).__init__(**kwargs)
        self.lstm_cell = lstm_cell
        self.dense = tf.keras.layers.Dense(output_size, activation=tf.nn.tanh)
        self.output_size = output_size
        self.state_size = [self.lstm_cell.state_size, output_size]

    def call(self, inputs, states, training=None):
        lstm_state, prev_output = states
        combined_inputs = tf.concat([inputs, prev_output], axis=-1)
        lstm_output, new_lstm_state = self.lstm_cell(combined_inputs, lstm_state, training=training)
        output = self.dense(lstm_output)
        return output, [new_lstm_state, output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initial_lstm_state = self.lstm_cell.get_initial_state(inputs, batch_size, dtype)
        initial_output = tf.zeros((batch_size, self.output_size), dtype=dtype)
        return [initial_lstm_state, initial_output]

# trainable initial input layer
class TrainableInitialInput(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(TrainableInitialInput, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # 初始化一个可训练的变量作为初始输入
        self.trainable_init_input = self.add_weight(
            shape=(1, 1, self.output_dim),
            initializer="random_normal",
            trainable=True,
            name='trainable_init_input'
        )

    def call(self, inputs):
        # 扩展初始输入以匹配输入批次的大小
        batch_size = tf.shape(inputs)[0]
        expanded_init_input = tf.tile(self.trainable_init_input, [batch_size, 1, 1])
        return expanded_init_input


# Other useful functions ########################################
def compute_sequence_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)

    return length


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant