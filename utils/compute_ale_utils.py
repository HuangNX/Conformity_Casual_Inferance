import numpy as np
import xarray as xr

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

def create_ale_dataset(ale_dict, quantiles, values, model_names, feature_name):
    """
    Function to Create an xarray.Dataset for Storing ALE Calculation Results
    Parameters:
    ale (np.ndarray): Array of ALE values with shape (n_bootstrap, n_bins).
    quantiles (np.ndarray): Array of feature quantiles with shape (n_bins,).
    values (np.ndarray): Array of feature values with shape either (n_bootstrap, n_bins, n_X) (3D) or (n_X) (2D).
    model_name (str): Name of the model.
    feature_name (str): Name of the feature.
    Returns:
    xr.Dataset: An xarray.Dataset containing the ALE calculation results.
    """

    # quantiles取平均
    quantiles = (quantiles[:-1] + quantiles[1:]) / 2

    # 如果 values 是三维数组，则展平为一维数组
    if values.ndim > 1:
        flattened_values = values.reshape(-1)
    else:
        flattened_values = values

    # 创建data_vars字典
    data_vars = {}
    for model_name in model_names:
        ale = ale_dict[model_name]
        # 如果 ale 是一维数组，则添加一个维度
        if ale.ndim == 1:
            ale = ale[np.newaxis, :]
        
        data_vars[f"{feature_name}__{model_name}__ale"] = (["n_bootstrap", f"n_bins__{feature_name}"], ale)

    data_vars[f"{feature_name}__bin_values"] = ([f"n_bins__{feature_name}"], quantiles)
    data_vars[feature_name] = (["n_X"], flattened_values)


    # 创建 xarray.Dataset
    dataset = xr.Dataset(
        data_vars=data_vars,
        #coords={
        #    "n_bootstrap": np.arange(ale.shape[0]),  # 根据 ale 的形状自动确定 n_bootstrap
        #    f"n_bins__{feature_name}": np.arange(quantiles.shape[0]),  # 根据 quantiles 的形状自动确定 n_bins
        #    "n_X": np.arange(flattened_values.shape[0]),  # 根据展平后的 values 形状确定 n_X
        #},
        attrs={
            "estimator_output": "raw",
            "estimators used": model_names,
            "method": "ale",
            "dimension": "1D",
            "features used": [feature_name],
        }
    )

    return dataset