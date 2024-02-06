import numpy as np
import pandas as pd

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

def get_dataset_splits(dataset, train_index, val_index, test_index, use_predicted_confounders):
    if use_predicted_confounders:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments',
                        'predicted_confounders', 'outcomes']
    else:
        dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']

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