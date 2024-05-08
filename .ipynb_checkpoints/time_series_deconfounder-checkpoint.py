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
import gc
'''
'0'：显示所有日志（默认设置）。
'1'：过滤掉INFO日志。
'2'：同时过滤掉INFO和WARNING日志。
'3'：过滤掉INFO、WARNING和ERROR日志，几乎不显示任何日志。
'''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import copy

from sklearn.model_selection import ShuffleSplit

from utils.evaluation_utils import save_data, write_results_to_file
from factor_model import FactorModel
from feature_engineer import Feature_Engineering
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
    
#     p_value = model.eval_predictive_checks(dataset_val)
#     plt.figure(figsize=(10, 6))
#     plt.plot(p_value, marker='o', linestyle='-')
#     plt.title('Predictive Checks P-Values Over Time')
#     plt.xlabel('Time')
#     plt.ylabel('P-Value')
#     plt.grid(True)

#     # 保存图像到本地文件
#     plt.savefig('../p_values_plot.png')
#     plt.close()  # 关闭图形，避免在 Jupyter 等环境中重复显示

    return predicted_confounders

def train_rmsn(dataset_map, model_name, b_use_predicted_confounders):
    model_name = model_name + '_use_confounders_' + str(b_use_predicted_confounders)
    MODEL_ROOT = os.path.join('results', model_name)

    if not os.path.exists(MODEL_ROOT):
        os.mkdir(MODEL_ROOT)
        print("Directory ", MODEL_ROOT, " Created ")
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
    FE_predict = Feature_Engineering(predict_data, for_factor_model = False)
    num_samples, length, num_covariates = predict_data['covariates'].shape
    _, _, num_treatments = predict_data['treatments'].shape

    FE_predict.get_normalize_params(predict_data, num_covariates, num_treatments)
    scale_params = FE_predict.scale_params
    predict_data['output_means'] = scale_params['outcomes'][:, 0]
    predict_data['output_stds'] = scale_params['outcomes'][:, 1]

    predict_data = FE_predict.get_dataset_normalize(predict_data, num_covariates, num_treatments)
    
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
    # config = {'covariate_cols':['gender','age','income','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4', 'weekday_5', 'weekday_6',
    #               'voluntary','festivals','year_0','year_1','year_2','edu','married','dist_yes','density', 'spatial_ent','temperal_ent','flow','temperature','percipit'],
    #          'treatment_cols':['conformity','E1','E2'],
    #          'confounder_cols':['confounder_{}'.format(i) for i in range(dataset['predicted_confounders'].shape[2])]}
    #config = {'covariate_cols':['gender','age','income','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4', 'weekday_5', 'weekday_6',
    #            'voluntary','festivals','year_0','year_1','year_2','density','landusemix','road_density','center','subway','edu',
    #            'married','dist_yes','temperature','percipit'],
    #        'treatment_cols':['conformity','E1','E2'],
    #        'confounder_cols':['confounder_{}'.format(i) for i in range(dataset['predicted_confounders'].shape[2])]
    #        }
    #config = {'covariate_cols':['gender','age','income','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4', 'weekday_5', 'weekday_6',
    #            'voluntary','festivals','year_0','year_1','year_2','density','edu',
    #            'married','dist_yes','temperature','percipit','E1','E2'],
    #        'treatment_cols':['conformity'],
    #        }
    config = {'covariate_cols':['gender','age','income','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4', 'weekday_5', 'weekday_6',
                'voluntary','festivals','year_0','year_1','year_2','density','edu',
                'married','dist_yes','temperature','percipit'],
            'treatment_cols':['conformity','E1','E2'],
            'confounder_cols':['confounder_{}'.format(i) for i in range(dataset['predicted_confounders'].shape[2])]
            }
    
    # shape of data
    num_samples, length, num_covariates = dataset['covariates'].shape
    _, _, num_treatments = dataset['treatments'].shape
    ## 先计算标准化的参数
    #scale_params = get_normalize_params(dataset, num_covariates, num_treatments)
    #dataset['output_means'] = scale_params['outcomes'][:, 0]
    #dataset['output_stds'] = scale_params['outcomes'][:, 1]
    
    ## construct dataframe
    #all_data = np.concatenate((dataset['covariates'].reshape(-1, len(config['covariate_cols'])), \
    #    dataset['treatments'].reshape(-1, len(config['treatment_cols']))), axis=1)
    #all_data = np.concatenate((all_data, dataset['predicted_confounders'].reshape(-1, len(config['confounder_cols']))), axis=1)
    #all_cols = config['covariate_cols'] + config['treatment_cols'] + config['confounder_cols']

    #X = pd.DataFrame(all_data, columns=all_cols)
    FE_predict = Feature_Engineering(dataset, for_factor_model = False)
    X = FE_predict.construct_dataframe(config)
    dataset['output_means'] = FE_predict.scale_params['outcomes'][:, 0]
    dataset['output_stds'] = FE_predict.scale_params['outcomes'][:, 1]
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
    def ale_use_predict(X, dataset, config, model_root, FE_predict, b_use_predicted_confounders):
        # change data
        mod_dataset = dataset.copy()
        # 将修改完的值添加到mod_dataset
        covariate_shapes = dataset['covariates'].shape
        mod_dataset['covariates'] = X[config['covariate_cols']].values.reshape(covariate_shapes[0],covariate_shapes[1],covariate_shapes[2])
        treatment_shapes = dataset['treatments'].shape
        mod_dataset['treatments'] = X[config['treatment_cols']].values.reshape(treatment_shapes[0],treatment_shapes[1],treatment_shapes[2])
        
        # 使用原来的参数做标准化
        mod_dataset = FE_predict.get_dataset_normalize(mod_dataset, num_covariates, num_treatments)
        # 调用函数预测
        predictions, _ = \
        rnn_predict(dataset=mod_dataset, MODEL_ROOT=model_root,
                    b_use_predicted_confounders=b_use_predicted_confounders)
        outputs = predictions.reshape(-1)
        del mod_dataset; gc.collect()
        return outputs
    ##############################################################################################################
    
    # define predictor
    predictor = lambda x: ale_use_predict(x, dataset, config, model_root, FE_predict, b_use_predicted_confounders)

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

    def obtain_dataset(dataset, column_index, selection=1):
        dataset_part = {}
        selected_indices = np.where(dataset['covariates'][:, :, column_index] == selection)[0]
        unique_selected_indices = np.unique(selected_indices)
        for key in dataset.keys():
            dataset_part[key] = dataset[key][unique_selected_indices]
        return dataset_part

    # sampling from big dataset
    def dataset_sampling(dataset, num_select=32000):
        num_samples = dataset['covariates'].shape[0]
        # Generate 32,000 random and unique indexes
        indices = np.random.choice(num_samples, num_select, replace=False)
        # Use these indexes to extract a corresponding sample from each array
        dataset = {key: value[indices] for key, value in dataset.items()}
        return dataset

    #选择2020年的数据单独跑一次
    # column_2020 = 13  # index of year 2020 indicator
    # dataset = obtain_dataset(dataset, column_2020)

    # 数据采样
    dataset = dataset_sampling(dataset, num_select=32000)
    
    #shuffle_split = ShuffleSplit(n_splits=1, test_size=0.1, random_state=10)
    #train_index, test_index = next(shuffle_split.split(dataset['covariates'][:, :, 0]))
    #shuffle_split = ShuffleSplit(n_splits=1, test_size=0.11, random_state=10)
    #train_index, val_index = next(shuffle_split.split(dataset['covariates'][train_index, :, 0]))
    
    # ############################# Factor Model #################################################################
    #FE = Feature_Engineering(dataset, for_factor_model=True)
    #dataset_map = FE.get_dataset_splits(train_index, val_index, test_index, use_predicted_confounders=False)
     
    #dataset_train = dataset_map['training_data']
    #dataset_val = dataset_map['validation_data']

    #logging.info("Fitting factor model")
    #predicted_confounders = train_factor_model(dataset_train, dataset_val,
    #                                      dataset,
    #                                      num_confounders=num_substitute_confounders,
    #                                      b_hyperparameter_optimisation=b_hyperparm_tuning,
    #                                      hyperparams_file=factor_model_hyperparams_file)
    ##print(predicted_confounders)
    #dataset['predicted_confounders'] = predicted_confounders
    ##write_results_to_file(dataset_with_confounders_filename, dataset)
    #save_data(dataset_with_confounders_filename, dataset)
    #logging.info('Finishing saving dataset with confounders!')

    # ############################# Recurrent Marginal Network #################################################################
    #FE = Feature_Engineering(dataset, for_factor_model=False)
    #dataset_map = FE.get_dataset_splits(train_index, val_index, test_index, use_predicted_confounders=False)

    #logging.info('Fitting counfounded recurrent marginal structural networks.')
    #train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False)

    #dataset_map = FE.get_dataset_splits(train_index, val_index, test_index, use_predicted_confounders=True)
    #logging.info(
    #'Fitting deconfounded (D_Z = {}) recurrent marginal structural networks.'.format(num_substitute_confounders))
    #train_rmsn(dataset_map, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True)
    
    #logging.info('Predicting treatment effects of conformity factor.')
    #results = predict_effects(dataset, 'rmsn_' + str(exp_name), calculate_counterfactual=False, b_use_predicted_confounders=True)
    #print(results['predictions'])
    #potential_results = predict_effects(dataset_map, 'rmsn_' + str(exp_name), calculate_counterfactual=True, b_use_predicted_confounders=True)
    # write_results_to_file(model_prediction_file, results)
    # logging.info("Successfully saved predictions. Finished.")
    
    # ############################# Accumulated Local Effects #################################################################
    # plot ale

    #split 3 years date [12-year2019, 13-year2020, 14-year2023]
    # column_2019 = 12  # index of year 2019 indicator
    # column_2020 = 13  # index of year 2020 indicator
    # column_2023 = 14  # index of year 2023 indicator
    ### 人群异质性
    ##column_gender = 0; column_age = 1; column_income = 2;

    # dataset19 = obtain_dataset(dataset, column_2019)
    # dataset20 = obtain_dataset(dataset, column_2020)
    # dataset23 = obtain_dataset(dataset, column_2023)

    #logging.info('Compute Single Accumulated Local Effects (ALE), draw and save the ale plot')
    #single_ale_fig, ale = compute_ale(dataset19, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False, features=['conformity'])
    #np.save("../data/result_data/raw_ale_2019_no_confounder.npy", ale)
    #single_ale_fig, ale = compute_ale(dataset20, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False, features=['conformity'])
    #np.save("../data/result_data/raw_ale_2020_no_confounder.npy", ale)
    #single_ale_fig, ale = compute_ale(dataset23, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False, features=['conformity'])
    #np.save("../data/result_data/raw_ale_2023_no_confounder.npy", ale)
    #single_ale_fig.savefig('results/image/sample_only_20_conformity_ale.png')
    #delete policy, see effect
    #dataset['treatments'][:, :, 1:] = 0

    # logging.info('Compute Time Windows Accumulated Local Effects (ALE), draw and save the ale plot')
    # window_ale_fig, ale2019 = compute_ale(dataset19, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False, features=['week', 'conformity'])
    # np.save("../data/result_data/raw_week_ale_2019_no_confounder.npy", ale2019.filled(np.nan))
    # window_ale_fig.savefig('results/image/all_sample_conformity_time_ale_2019_add_ent.png')
    # window_ale_fig, ale2020 = compute_ale(dataset, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['week', 'conformity'])
    #np.save("../data/result_data/raw_week_ale_2020_no_confounder.npy", ale2020.filled(np.nan))
    # window_ale_fig.savefig('results/image/sample_only_20_conformity_policy_ale.png')
    # window_ale_fig, ale2023 = compute_ale(dataset23, 'rmsn_' + str(exp_name), b_use_predicted_confounders=False, features=['week', 'conformity'])
    # np.save("../data/result_data/raw_week_ale_2023_no_confounder.npy", ale2023.filled(np.nan))
    # window_ale_fig.savefig('results/image/all_sample_conformity_time_ale_2023_add_ent.png')
    #收入异质性
    #for income_level in range(5):
    #    dataset_subincome = obtain_dataset(dataset, column_income, selection=income_level)
    #    dataset19 = obtain_dataset(dataset_subincome, column_2019)
    #    single_ale_fig, ale = compute_ale(dataset19, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity'])
    #    np.save("../data/raw_ale_19_income_{}.npy".format(income_level), ale)
    #    dataset20 = obtain_dataset(dataset_subincome, column_2020)
    #    single_ale_fig, ale = compute_ale(dataset20, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity'])
    #    np.save("../data/raw_ale_20_income_{}.npy".format(income_level), ale)
    #    dataset23 = obtain_dataset(dataset_subincome, column_2023)
    #    single_ale_fig, ale = compute_ale(dataset23, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity'])
    #    np.save("../data/raw_ale_23_income_{}.npy".format(income_level), ale)

    logging.info('Compute Interactive Accumulated Local Effects (ALE), draw and save the ale plot')
    interact_ale_fig, ale = compute_ale(dataset, 'rmsn_' + str(exp_name), b_use_predicted_confounders=True, features=['conformity','voluntary'])
    interact_ale_fig.savefig('results/image/sample_only_20_conformity_case_ale.png')
