'''
Title: Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders
Authors: Ioana Bica, Ahmed M. Alaa, Mihaela van der Schaar
International Conference on Machine Learning (ICML) 2020

Last Updated Date: July 20th 2020
Code Author: Ioana Bica (ioana.bica95@gmail.com)
'''
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import copy

from sklearn.model_selection import ShuffleSplit

from utils.evaluation_utils import save_data, write_results_to_file
from factor_model import FactorModel
from rmsn.script_rnn_fit import rnn_fit
from rmsn.script_propensity_generation import propensity_generation
from rmsn.script_rnn_predict import rnn_predict
from ale import ale_plot


def train_factor_model(dataset_train, dataset_val, dataset, num_confounders, hyperparams_file,
                       b_hyperparameter_optimisation):
    _, length, num_covariates = dataset_train['covariates'].shape
    num_treatments = dataset_train['treatments'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_confounders': num_confounders,
              'max_sequence_length': length,
              'num_epochs': 100}

    hyperparams = dict()
    num_simulations = 50
    best_validation_loss = 100
    if b_hyperparameter_optimisation:
        logging.info("Performing hyperparameter optimization")
        for simulation in range(num_simulations):
            logging.info("Simulation {} out of {}".format(simulation + 1, num_simulations))

            hyperparams['rnn_hidden_units'] = np.random.choice([32, 64, 128, 256])
            hyperparams['fc_hidden_units'] = np.random.choice([32, 64, 128])
            hyperparams['learning_rate'] = np.random.choice([0.01, 0.001, 0.0001])
            hyperparams['batch_size'] = np.random.choice([64, 128, 256])
            hyperparams['rnn_keep_prob'] = np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9])

            logging.info("Current hyperparams used for training \n {}".format(hyperparams))
            model = FactorModel(params, hyperparams)
            model.train(dataset_train, dataset_val)
            validation_loss = model.eval_network(dataset_val)

            if (validation_loss < best_validation_loss):
                logging.info(
                    "Updating best validation loss | Previous best validation loss: {} | Current best validation loss: {}".format(
                        best_validation_loss, validation_loss))
                best_validation_loss = validation_loss
                best_hyperparams = hyperparams.copy()

            logging.info("Best hyperparams: \n {}".format(best_hyperparams))

        write_results_to_file(hyperparams_file, best_hyperparams)

    else:
        best_hyperparams = {
            'rnn_hidden_units': 128,
            'fc_hidden_units': 128,
            'learning_rate': 0.001,
            'batch_size': 128,
            'rnn_keep_prob': 0.8}

    model = FactorModel(params, best_hyperparams)
    model.train(dataset_train, dataset_val)
    predicted_confounders = model.compute_hidden_confounders(dataset)
    
    #p_value = model.eval_predictive_checks(dataset_val)
    #plt.figure(figsize=(10, 6))
    #plt.plot(p_value, marker='o', linestyle='-')
    #plt.title('Predictive Checks P-Values Over Time')
    #plt.xlabel('Time')
    #plt.ylabel('P-Value')
    #plt.grid(True)

    ## 保存图像到本地文件
    #plt.savefig('../p_values_plot.png')
    #plt.close()  # 关闭图形，避免在 Jupyter 等环境中重复显示

    return predicted_confounders

def compute_mean_std(column):
    if np.all((column == 0) | (column == 1)) or len(np.unique(column)) == 1:
        mean = 0
        std = 1
    else:
        mean = np.mean(column)
        std = np.std(column)
    return mean, std

def get_normalize_params(dataset, num_covariates, num_treatments):
    scale_params = dict()

    for key in ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']:
        scale_params[key] = []

    for covariate_id in range(num_covariates):

        column = dataset['previous_covariates'][:, :, covariate_id]
        # if feature is [0,1], then mean = 0, std = 1
        pre_covariate_mean, pre_covariate_std = compute_mean_std(column)
        scale_params['previous_covariates'].append(np.array([pre_covariate_mean, pre_covariate_std]))

        column = dataset['covariates'][:, :, covariate_id]
        covariate_mean, covariate_std = compute_mean_std(column)
        scale_params['covariates'].append(np.array([covariate_mean, covariate_std]))

    for treatment_id in range(num_treatments):
        column = dataset['previous_treatments'][:, :, treatment_id]
        pre_treatment_mean, pre_treatment_std = compute_mean_std(column)
        scale_params['previous_treatments'].append(np.array([pre_treatment_mean, pre_treatment_std]))

        column = dataset['treatments'][:, :, treatment_id]
        treatment_mean, treatment_std = compute_mean_std(column)
        scale_params['treatments'].append(np.array([treatment_mean, treatment_std]))

    scale_params['outcomes'].append(np.array([np.mean(dataset['outcomes']), np.std(dataset['outcomes'])]))

    for key in scale_params.keys():
        scale_params[key] = np.array(scale_params[key])

    return scale_params

def get_dataset_normalize(dataset, scale_params, num_covariates, num_treatments):
    for covariate_id in range(num_covariates):
        dataset['previous_covariates'][:, :, covariate_id] = \
            (dataset['previous_covariates'][:, :, covariate_id] - scale_params['previous_covariates'][covariate_id, 0]) / \
            scale_params['previous_covariates'][covariate_id, 1]

        dataset['covariates'][:, :, covariate_id] = \
            (dataset['covariates'][:, :, covariate_id] - scale_params['covariates'][covariate_id, 0]) / \
            scale_params['covariates'][covariate_id, 1]

    for treatment_id in range(num_treatments):
        dataset['previous_treatments'][:, :, treatment_id] = \
            (dataset['previous_treatments'][:, :, treatment_id] - scale_params['previous_treatments'][treatment_id, 0]) / \
            scale_params['previous_treatments'][treatment_id, 1]

        dataset['treatments'][:, :, treatment_id] = \
            (dataset['treatments'][:, :, treatment_id] - scale_params['treatments'][treatment_id, 0]) / \
            scale_params['treatments'][treatment_id, 1]

    dataset['outcomes'] = (dataset['outcomes'] - scale_params['outcomes'][0,0]) /scale_params['outcomes'][0,1]

    return dataset

# 感觉这个名字应该要换，因为不只是在做数据集的划分
def get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

    # data type transformation
    for key in dataset.keys():
        if key!='sequence_length':
            dataset[key] = dataset[key].astype(np.float32)

    # dataset split
    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index, :, :]
        dataset_val[key] = dataset[key][val_index, :, :]
        dataset_test[key] = dataset[key][test_index, :, :]

    _, length, num_covariates = dataset_train['covariates'].shape
    _, _, num_treatments = dataset_train['treatments'].shape

    # normalization
    scale_params = get_normalize_params(dataset_train, num_covariates, num_treatments) 
    dataset_train['output_means'] = scale_params['outcomes'][:, 0]
    dataset_train['output_stds'] = scale_params['outcomes'][:, 1]

    dataset_train = get_dataset_normalize(dataset_train, scale_params, num_covariates, num_treatments)
    dataset_val = get_dataset_normalize(dataset_val, scale_params, num_covariates, num_treatments)
    dataset_test = get_dataset_normalize(dataset_test, scale_params, num_covariates, num_treatments)

    key = 'sequence_length'
    dataset_train[key] = dataset[key][train_index]
    dataset_val[key] = dataset[key][val_index]
    dataset_test[key] = dataset[key][test_index]

    dataset_map = dict()

    dataset_map['num_time_steps'] = length
    dataset_map['training_data'] = dataset_train
    dataset_map['validation_data'] = dataset_val
    dataset_map['test_data'] = dataset_test

    return dataset_map


def train_rmsn(dataset_map, model_name, b_use_predicted_confounders):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    #if not os.path.exists(MODEL_ROOT):
    #    os.mkdir(MODEL_ROOT)
    #    print("Directory ", MODEL_ROOT, " Created ")
    #else:
    #    # Need to delete previously saved model.
    #    shutil.rmtree(MODEL_ROOT)
    #    os.mkdir(MODEL_ROOT)
    #    print("Directory ", MODEL_ROOT, " Created ")

    #rnn_fit(dataset_map=dataset_map, networks_to_train='propensity_networks', MODEL_ROOT=MODEL_ROOT,
    #        b_use_predicted_confounders=b_use_predicted_confounders)

    propensity_generation(dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
                           b_use_predicted_confounders=b_use_predicted_confounders)

    rnn_fit(networks_to_train='encoder', dataset_map=dataset_map, MODEL_ROOT=MODEL_ROOT,
            b_use_predicted_confounders=b_use_predicted_confounders)

#     rmse = np.sqrt(np.mean(rmsn_mse)) * 100
    #return rmsn_mse[list(rmsn_mse)[0]]

def predict_effects(predict_data, model_name, calculate_counterfactual, b_use_predicted_confounders, treatment_index=None):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    # modify the interested treatment in dataset_map to 0 
    if calculate_counterfactual:
        predict_data['treatments'][:, :, treatment_index] = 0
        
    # dataset['covariates'] = dataset['covariates'][:,5:,:]
    # dataset['treatments'] = dataset['treatments'][:,5:,:]
    # dataset['predicted_confounders'] = dataset['predicted_confounders'][:,5:,:]
    # dataset['outcomes'] = dataset['outcomes'][:,5:,:]
    # dataset['sequence_length']-=5
    
    # data preprocessing - normalization
    num_samples, length, num_covariates = predict_data['covariates'].shape
    _, _, num_treatments = predict_data['treatments'].shape

    scale_params = get_normalize_params(predict_data, num_covariates, num_treatments) 
    predict_data['output_means'] = scale_params['outcomes'][:, 0]
    predict_data['output_stds'] = scale_params['outcomes'][:, 1]

    predict_data = get_dataset_normalize(predict_data, scale_params, num_covariates, num_treatments)
    
    # propensity_generation(dataset_map=dataset, MODEL_ROOT=MODEL_ROOT,
    #                       b_use_predicted_confounders=b_use_predicted_confounders, b_use_all_data=True)
    
    predictions, observations = \
        rnn_predict(dataset=predict_data, MODEL_ROOT=MODEL_ROOT,
                    b_use_predicted_confounders=b_use_predicted_confounders)

    print(predictions)
    print(f"shape of predictions = {predictions.shape}")
    print(observations)
    print(f"shape of observations = {observations.shape}")
    results = dict()
    results['predictions'] = predictions
    results['observations'] = observations

    return results


def compute_ale(dataset, model_name, b_use_predicted_confounders, features):
    # compute accumulated local effects (ALE)
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    model_root = os.path.join('results', model_name)
    config = {'covariate_cols':['gender','age','income','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4', 'weekday_5', 'weekday_6',
                  'voluntary','festivals','year_0','year_1','year_2','density','landusemix','road_density','center','subway','edu',
                  'married','dist_yes','temperature','percipit'],
              'treatment_cols':['conformity','E1','E2'],
              'confounder_cols':['confounder_{}'.format(i) for i in range(dataset['predicted_confounders'].shape[2])]}
    # config = {'covariate_cols':['gender','age','income3','weekday_0','weekday_1','weekday_2','weekday_3',\
    #                       'weekday_4', 'weekday_5', 'weekday_6','sprtransp','spring','precip', 'voluntary'],
    #           'treatment_cols':['conformity','restrict','open'],
    #           'confounder_cols':['confounder_{}'.format(i) for i in range(dataset['predicted_confounders'].shape[2])]}
    
    # shape of data
    num_samples, length, num_covariates = dataset['covariates'].shape
    _, _, num_treatments = dataset['treatments'].shape
    # 先计算标准化的参数
    scale_params = get_normalize_params(dataset, num_covariates, num_treatments) 
    dataset['output_means'] = scale_params['outcomes'][:, 0]
    dataset['output_stds'] = scale_params['outcomes'][:, 1]
    
    # construct dataframe
    all_data = np.concatenate((dataset['covariates'].reshape(-1, len(config['covariate_cols'])), \
        dataset['treatments'].reshape(-1, len(config['treatment_cols']))), axis=1)
    all_data = np.concatenate((all_data, dataset['predicted_confounders'].reshape(-1, len(config['confounder_cols']))), axis=1)
    all_cols = config['covariate_cols'] + config['treatment_cols'] + config['confounder_cols']

    X = pd.DataFrame(all_data, columns=all_cols)
    # add timeline and week column
    X['timeline'] = np.tile(np.arange(length), num_samples)
    # start_date = pd.Timestamp('2020-01-01')
    # X['week'] = X['timeline'].apply(lambda x: start_date + pd.Timedelta(days=x)).dt.week - 1
    # 20年的数据是从2019.12.21开始的，直接按7天编码
    start_date = pd.Timestamp('2019-12-21')
    X['date'] = X['timeline'].apply(lambda x: start_date + pd.Timedelta(days=x))
    using_policy_period = True
    if using_policy_period:
        # 按政策时期分类
        # Define the dates to categorize
        categorization_dates = pd.to_datetime(
            ['2020-01-19','2020-01-23', '2020-02-07', '2020-02-10', '2020-02-17', '2020-02-23', '2020-03-21', '2020-03-29', '2020-05-09']
        )
        # Function to categorize dates
        def categorize_date(row_date):
            for i, cat_date in enumerate(categorization_dates):
                if row_date < cat_date:
                    return i
            return len(categorization_dates)
        X['week'] = X['date'].apply(categorize_date)
    else:
         X['week'] = ((X['date'] - start_date).dt.days / 7).astype(int)
    
    # # 确保周编号在年末时正确地重置
    # X['year'] = X['timeline'].apply(lambda x: start_date + pd.Timedelta(days=x)).dt.year
    # X['week'] = X.apply(lambda row: row['week'] + 52* (row['year'] - 2020), axis=1) # mistake

    ##############################################################################################################
    def ale_use_predict(X, dataset, config, model_root, scale_params, b_use_predicted_confounders):
        # change data
        mod_dataset = dataset.copy()
        # 将修改完的值添加到mod_dataset
        covariate_shapes = dataset['covariates'].shape
        mod_dataset['covariates'] = X[config['covariate_cols']].values.reshape(covariate_shapes[0],covariate_shapes[1],covariate_shapes[2])
        treatment_shapes = dataset['treatments'].shape
        mod_dataset['treatments'] = X[config['treatment_cols']].values.reshape(treatment_shapes[0],treatment_shapes[1],treatment_shapes[2])
        
        # 使用原来的参数做标准化
        mod_dataset = get_dataset_normalize(mod_dataset, scale_params, num_covariates, num_treatments)
        # 调用函数预测
        predictions, _ = \
        rnn_predict(dataset=mod_dataset, MODEL_ROOT=model_root,
                    b_use_predicted_confounders=b_use_predicted_confounders)
        outputs = predictions.reshape(-1)
        return outputs
    ##############################################################################################################
    
    # define predictor
    predictor = lambda x: ale_use_predict(x, dataset, config, model_root, scale_params, b_use_predicted_confounders)

    # Plots ALE function of specified features based on training set.
    # if len(features)==1:
    #     ale_fig, ale_ax = ale_plot(
    #                 None, 
    #                 X,
    #                 features,
    #                 bins=20,
    #                 predictor=predictor,
    #                 monte_carlo=True,
    #                 monte_carlo_rep=100,
    #                 monte_carlo_ratio=0.6,)
    # else:
    ale_fig, ale_ax, ale = ale_plot(
                None, 
                X,
                features,
                bins=20,
                predictor=predictor,)

    return ale_fig, ale

def test_time_series_deconfounder(dataset, num_substitute_confounders, exp_name, dataset_with_confounders_filename,
                                  factor_model_hyperparams_file, model_prediction_file, b_hyperparm_tuning=False):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    #dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=False)
     
    #dataset_train = dataset_map['training_data']
    #dataset_val = dataset_map['validation_data']

    #logging.info("Fitting factor model")
    #predicted_confounders = train_factor_model(dataset_train, dataset_val,
    #                                        dataset,
    #                                        num_confounders=num_substitute_confounders,
    #                                        b_hyperparameter_optimisation=b_hyperparm_tuning,
    #                                        hyperparams_file=factor_model_hyperparams_file)
    ##print(predicted_confounders)
    #dataset['predicted_confounders'] = predicted_confounders
    ##write_results_to_file(dataset_with_confounders_filename, dataset)
    #save_data(dataset_with_confounders_filename, dataset)
    #logging.info('Finishing saving dataset with confounders!')

    dataset_map = get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders=True)

    #logging.info('Fitting counfounded recurrent marginal structural networks.')
    #rmse_without_confounders = train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False)

    logging.info(
        'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(num_substitute_confounders))
    train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True)

    # print("Outcome model RMSE when trained WITHOUT the hidden confounders.")
    # print(rmse_without_confounders)

    #print("Outcome model RMSE when trained WITH the substitutes for the hidden confounders.")
    #print(rmse_with_confounders)
    
    #logging.info('Predicting treatment effects of conformity factor.')
    #results = predict_effects(dataset, 'rmsn_' + str(exp_name), calculate_counterfactual=False, b_use_predicted_confounders=False)
    #potential_results = predict_effects(dataset_map, 'rmsn_' + str(exp_name), calculate_counterfactual=True, b_use_predicted_confounders=True)
    # write_results_to_file(model_prediction_file, results)
    # logging.info("Successfully saved predictions. Finished.")
    
    # plot ale
    # using only 2020 data [12-year2019, 13-year2020, 14-year2023]
    #column_index = 13  # index of year 2020 indicator
    #selected_indices = np.where(dataset['covariates'][:, :, column_index] == 1)[0]
    #unique_selected_indices = np.unique(selected_indices)
    #for key in dataset.keys():
    #    dataset[key] = dataset[key][unique_selected_indices]
        
    #logging.info('Compute Single Accumulated Local Effects (ALE), draw and save the ale plot')
    ##dataset1 = copy.deepcopy(dataset)
    #single_ale_fig, ale = compute_ale(dataset, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity'])
    #single_ale_fig.savefig('results/image/new_sample_1000_conformity_ale.png')

    #logging.info('Compute Time Windows Accumulated Local Effects (ALE), draw and save the ale plot')
    #dataset2 = copy.deepcopy(dataset)
    #window_ale_fig, ale = compute_ale(dataset2, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['week', 'conformity'])
    #window_ale_fig.savefig('results/image/new_sample_1000_conformity_policy_ale.png')

    #logging.info('Compute Interactive Accumulated Local Effects (ALE), draw and save the ale plot')
    #interact_ale_fig, ale = compute_ale(dataset, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity','voluntary'])
    #interact_ale_fig.savefig('results/image/new_sample_8000_conformity_case_ale.png')
