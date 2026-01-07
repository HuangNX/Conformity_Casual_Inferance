from h5py._hl.dataset import sel
import numpy as np
import pandas as pd

class Feature_Engineering:
    def __init__(self, dataset, for_factor_model):
        self.dataset = dataset
        self.for_factor_model = for_factor_model
        # treatment process
        #if not self.for_factor_model:
        #    bins = 20
        #    self.dataset['treatments'] = self.treatment_discretized(self.dataset['treatments'], bins)

    def compute_mean_std(self, column):
        if np.all((column == 0) | (column == 1)) or len(np.unique(column)) == 1:
            mean = 0
            std = 1
        else:
            mean = np.mean(column)
            std = np.std(column)
        return mean, std

    def get_normalize_params(self, dataset, num_covariates, num_treatments):
        self.scale_params = dict()

        for key in ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']:
            self.scale_params[key] = []

        for covariate_id in range(num_covariates):
            if self.for_factor_model:
                column = dataset['previous_covariates'][:, :, covariate_id]
                # if feature is [0,1], then mean = 0, std = 1
                pre_covariate_mean, pre_covariate_std = self.compute_mean_std(column)
                self.scale_params['previous_covariates'].append(np.array([pre_covariate_mean, pre_covariate_std]))

            column = dataset['covariates'][:, :, covariate_id]
            covariate_mean, covariate_std = self.compute_mean_std(column)
            self.scale_params['covariates'].append(np.array([covariate_mean, covariate_std]))
        
        for treatment_id in range(num_treatments):
            if self.for_factor_model:
                column = dataset['previous_treatments'][:, :, treatment_id]
                pre_treatment_mean, pre_treatment_std = self.compute_mean_std(column)
                self.scale_params['previous_treatments'].append(np.array([pre_treatment_mean, pre_treatment_std]))

                column = dataset['treatments'][:, :, treatment_id]
                treatment_mean, treatment_std = self.compute_mean_std(column)
                self.scale_params['treatments'].append(np.array([treatment_mean, treatment_std]))
            else:
                self.scale_params['previous_treatments'].append(np.array([0, 1]))
                self.scale_params['treatments'].append(np.array([0, 1]))

        self.scale_params['outcomes'].append(np.array([np.mean(dataset['outcomes']), np.std(dataset['outcomes'])]))

        for key in self.scale_params.keys():
            self.scale_params[key] = np.array(self.scale_params[key])

    def get_dataset_normalize(self, dataset, num_covariates, num_treatments):
        for covariate_id in range(num_covariates):
            if self.for_factor_model:
                dataset['previous_covariates'][:, :, covariate_id] = \
                    (dataset['previous_covariates'][:, :, covariate_id] - self.scale_params['previous_covariates'][covariate_id, 0]) / \
                    self.scale_params['previous_covariates'][covariate_id, 1]

            dataset['covariates'][:, :, covariate_id] = \
                (dataset['covariates'][:, :, covariate_id] - self.scale_params['covariates'][covariate_id, 0]) / \
                self.scale_params['covariates'][covariate_id, 1]

        for treatment_id in range(num_treatments):
            if self.for_factor_model:
                dataset['previous_treatments'][:, :, treatment_id] = \
                    (dataset['previous_treatments'][:, :, treatment_id] - self.scale_params['previous_treatments'][treatment_id, 0]) / \
                    self.scale_params['previous_treatments'][treatment_id, 1]

            dataset['treatments'][:, :, treatment_id] = \
                (dataset['treatments'][:, :, treatment_id] - self.scale_params['treatments'][treatment_id, 0]) / \
                self.scale_params['treatments'][treatment_id, 1]

        dataset['outcomes'] = (dataset['outcomes'] - self.scale_params['outcomes'][0,0]) /self.scale_params['outcomes'][0,1]

        return dataset

    def treatment_discretized(self, treatments, bins):
        continuous_indices = [index for index in range(treatments.shape[2]) if not np.all(np.isin(treatments[:, :, index], [0, 1]))]
        for index in continuous_indices:
            # Box coding is performed for each successive value column
            treatments[:, :, index] = np.digitize(treatments[:, :, index], 
                                                  bins=np.linspace(treatments[:, :, index].min(), treatments[:, :, index].max(), bins+1)[1:-1], 
                                                  right=True)
        treatments[:, :, index] = treatments[:, :, index].astype(np.int32)
        return treatments

    def get_dataset_splits(self, train_index, val_index, test_index, use_predicted_confounders):
        self.use_predicted_confounders = use_predicted_confounders
        if self.for_factor_model:
            dataset_keys = ['previous_covariates', 'previous_treatments', 'covariates', 'treatments', 'outcomes']
        else:
            if self.use_predicted_confounders:
                dataset_keys = ['covariates', 'treatments','predicted_confounders', 'outcomes']
            else:
                dataset_keys = ['covariates', 'treatments', 'outcomes']

        dataset_train = dict()
        dataset_val = dict()
        dataset_test = dict()
        for key in dataset_keys:
            dataset_train[key] = self.dataset[key][train_index, :, :]
            dataset_val[key] = self.dataset[key][val_index, :, :]
            dataset_test[key] = self.dataset[key][test_index, :, :]

        _, length, num_covariates = dataset_train['covariates'].shape
        _, _, num_treatments = dataset_train['treatments'].shape

        # normalization
        self.get_normalize_params(dataset_train, num_covariates, num_treatments) 
        dataset_train['output_means'] = self.scale_params['outcomes'][:, 0]
        dataset_train['output_stds'] = self.scale_params['outcomes'][:, 1]

        dataset_train = self.get_dataset_normalize(dataset_train, num_covariates, num_treatments)
        dataset_val = self.get_dataset_normalize(dataset_val, num_covariates, num_treatments)
        dataset_test = self.get_dataset_normalize(dataset_test, num_covariates, num_treatments)

        key = 'sequence_length'
        dataset_train[key] = self.dataset[key][train_index]
        dataset_val[key] = self.dataset[key][val_index]
        dataset_test[key] = self.dataset[key][test_index]

        dataset_map = dict()

        dataset_map['num_time_steps'] = length
        dataset_map['training_data'] = dataset_train
        dataset_map['validation_data'] = dataset_val
        dataset_map['test_data'] = dataset_test

        return dataset_map

    def construct_dataframe(self, config):
         # shape of data
        num_samples, length, num_covariates = self.dataset['covariates'].shape
        _, _, num_treatments = self.dataset['treatments'].shape
        # 先计算标准化的参数
        self.get_normalize_params(self.dataset, num_covariates, num_treatments)
        #dataset['output_means'] = scale_params['outcomes'][:, 0]
        #dataset['output_stds'] = scale_params['outcomes'][:, 1]

        # construct dataframe
        all_data = np.concatenate((self.dataset['covariates'].reshape(-1, len(config['covariate_cols'])), \
            self.dataset['treatments'].reshape(-1, len(config['treatment_cols']))), axis=1)
        if 'confounder_cols' in config:
            all_data = np.concatenate((all_data, self.dataset['predicted_confounders'].reshape(-1, len(config['confounder_cols']))), axis=1)
            all_cols = config['covariate_cols'] + config['treatment_cols'] + config['confounder_cols']
        else:
            all_cols = config['covariate_cols'] + config['treatment_cols']

        X = pd.DataFrame(all_data, columns=all_cols)

        return X

