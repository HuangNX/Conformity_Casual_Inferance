import numpy as np

def compute_quartiles(dataset, column_index):
    flow_values = dataset['covariates'][:, :, column_index].flatten()
    quartiles = np.percentile(flow_values, [25, 50, 75])
    return quartiles

def obtain_dataset_parts(dataset, column_index):
    quartiles = compute_quartiles(dataset, column_index)
    flow_values = dataset['covariates'][:,:, column_index]
    levels = np.zeros(flow_values.shape, dtype=int)
    levels[flow_values > quartiles[0]] = 1
    levels[flow_values > quartiles[1]] = 2
    levels[flow_values > quartiles[2]] = 3
    dataset_parts = {}
    for level in range(4):
        selected_indices = np.where(levels == level)[0]
        unique_selected_indices = np.unique(selected_indices)
        dataset_part = {}
        for key in dataset.keys():
            dataset_part[key] = dataset[key][unique_selected_indices]
        dataset_parts[level] = dataset_part
    return dataset_parts