from user_preprocessing.utils import process_adjacency_matrix, process_data
import user_preprocessing

import os
import json
import numpy as np

from user_preprocessing.loader import load_dataset

def preprocess(self, args, datafile, result_file):

    train_data, valid_data, test_data = load_dataset(
        datafile, args.train_length, args.valid_length, args.test_length)

    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        norm_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        norm_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        norm_statistic = None

    if norm_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(norm_statistic, f)

    x_train, y_train = process_data(train_data, args.window_size, args.horizon)
    x_valid, y_valid = process_data(valid_data, args.window_size, args.horizon)
    x_test, y_test = process_data(test_data, args.window_size, args.horizon)

    train_scaler = user_preprocessing.loader.CustomStandardScaler(mean=x_train.mean(), std=x_train.std())

    train_loader = user_preprocessing.loader.CustomSimpleDataLoader(train_scaler.transform(x_train),
                                                                train_scaler.transform(y_train), args.batch_size)
    valid_loader = user_preprocessing.loader.CustomSimpleDataLoader(train_scaler.transform(x_valid),
                                                                train_scaler.transform(y_valid), args.batch_size)
    
    test_scaler = user_preprocessing.loader.CustomStandardScaler(mean=x_test.mean(), std=x_test.std())

    test_loader = user_preprocessing.loader.CustomSimpleDataLoader(test_scaler.transform(x_test), 
                                                                test_scaler.transform(y_test), args.batch_size)
    return train_loader, valid_loader, test_loader, train_scaler, test_scaler