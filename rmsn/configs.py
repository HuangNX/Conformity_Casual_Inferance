"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018

Implementation of Recurrent Marginal Structural Networks (R-MSNs):
Brian Lim, Ahmed M Alaa, Mihaela van der Schaar, "Forecasting Treatment Responses Over Time Using Recurrent
Marginal Structural Networks", Advances in Neural Information Processing Systems, 2018.
"""

import os
import logging
import rmsn.libs.net_helpers as helpers
import numpy as np
import tensorflow as tf

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, "results")

# setup gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Optionally set memory growth on the selected GPU
        #tf.config.set_visible_devices(gpus[0], 'GPU')
        #tf.config.experimental.set_memory_growth(gpus[0], True)
        #logging.info("Using GPU 0 with memory growth")

        # set TensorFlow to use all GPU
        tf.config.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:
            # set GPU memery growth
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Using GPU with memory growth")
    except RuntimeError as e:
        # Changing device settings after the program is running may cause errors
        logging.info(e)
else:
    # if no GPU，using CPU
    logging.info("No GPU found, using CPU")

#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
strategy = tf.distribute.MirroredStrategy()

def get_parameters_from_string(serialisation_string):

    spec = serialisation_string.split("_")
    dropout_rate = float(spec[0])
    hidden_layer_size = int(spec[1])
    num_epochs = int(spec[2])
    minibatch_size = int(spec[3])
    learning_rate = float(spec[4])
    max_norm = float(spec[5])

    return (dropout_rate, hidden_layer_size, num_epochs, minibatch_size, learning_rate, max_norm)


def load_optimal_parameters(net_name, expt_name, MODEL_ROOT, select_max=False, add_net_name=False):

    model_folder = os.path.join(MODEL_ROOT, net_name)

    hyperparam_df = helpers.load_hyperparameter_results(model_folder, net_name)
    print (model_folder)
    print (net_name)
    print (hyperparam_df)

    validation_scores = hyperparam_df.loc["validation_loss"]

    # Select optimal
    if select_max:
        best_score = validation_scores.max()
    else:
        best_score = validation_scores.min()

    names = np.array(validation_scores.index)
    best_expt = names[validation_scores==best_score][0]

    serialisation_string = best_expt.replace(expt_name+"_", "")
    params = get_parameters_from_string(serialisation_string)

    if add_net_name:
        params = [net_name] + list(params)

    return params


# In[*]: Test Stuff:
if __name__ == "__main__":
    expt_name = "treatment_effects"
    net_name = "treatment_rnn_action_inputs_only"


    test = load_optimal_parameters(net_name, expt_name, select_max=True)


