import numpy as np
import pickle
import h5py
import os


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)

def load_data_from_file(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def save_dict_to_hdf5(path, data, suffix='/'):
    with h5py.File(path, 'w') as f:
        for key, value in data.items():
            if isinstance(value, (np.ndarray, np.int64, np.float64, str, int, float)):
                f[suffix + key] = value
            elif isinstance(value, dict):
                sub_group = f.create_group(suffix + key)
                save_dict_to_hdf5(value, f, path + key + '/')
            else:
                raise ValueError(f"Unsupported data type for key: {key}")
    print(f"Finish {path}!")

def load_dict_from_hdf5(path, suffix='/'):
    result = {}
    with h5py.File(path, 'r') as f:
        for key, item in f[suffix].items():
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = load_dict_from_hdf5(f, suffix + key + '/')
    return result

def save_data(filename, data):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.h5':
        save_dict_to_hdf5(filename, data)
    elif file_extension == '.txt':
        write_results_to_file(filename, data)
    else:
        raise ValueError("Invalid file extension! Save failed")

def load_data(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.h5':
        return load_dict_from_hdf5(filename)
    elif file_extension == '.txt':
        return load_data_from_file(filename)
    else:
        raise ValueError("Invalid file extension! Load failed")

