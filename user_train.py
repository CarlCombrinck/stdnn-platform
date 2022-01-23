import user_preprocessing
from user_preprocessing.utils import process_data

# TODO Move to another module (for decorators)
from stdnn.models.utils import timed

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os
import time

# @timed(operation_name="Train")
def train_model(self, train_data, valid_data, args, result_file):
    """
    Trains a graph neural network model and returns a set of validation performance metrics

    Parameters
    ----------
    train_data : numpy.ndarray
        Train set
    valid_data : numpy.ndarray
        Validation set
    args : argparse.Namespace
        Command line arguments
    result_file : str
        Directory to store trained model parameter files

    Returns
    -------
    dict
    """
    self.model.to(args.get("device"))
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.get("norm_method") == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        norm_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}

    elif args.get("norm_method") == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        norm_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        norm_statistic = None
    if norm_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(norm_statistic, f)

    if args.get("optimizer") == 'RMSProp':
        optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=args.get("lr"), eps=1e-08)
    elif args.get("optimizer") == 'SGD':
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
    elif args.get("optimizer") == 'Adagrad':
        optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
    elif args.get("optimizer") == 'Adadelta':
        optimizer = torch.optim.Adadelta(params=self.model.parameters(), lr=args.get("lr"), weight_decay=args.get("weight_decay"))
    else:
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.get("lr"), betas=(0.9, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.get("decay_rate"))
    scaler = None

    x_train, y_train = process_data(train_data, args.get("window_size"), args.get("horizon"))
    x_valid, y_valid = process_data(valid_data, args.get("window_size"), args.get("horizon"))

    scaler = user_preprocessing.loader.CustomStandardScaler(mean=x_train.mean(), std=x_train.std())

    train_loader = user_preprocessing.loader.CustomSimpleDataLoader(scaler.transform(x_train),
                                                                scaler.transform(y_train), args.get("batch_size"))
    valid_loader = user_preprocessing.loader.CustomSimpleDataLoader(scaler.transform(x_valid),
                                                                scaler.transform(y_valid), args.get("batch_size"))

    criterion = nn.MSELoss(reduction='mean').to(args.get("device"))

    total_params = 0
    for name, parameter in self.model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    # print(f"Total Trainable Parameters: {total_params}")
    # print("Model: GWN")
    # print()

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    valid_frame = pd.DataFrame(columns=["epoch", "mape", "mae", "rmse"])
    for epoch in range(args.get("epoch")):
        epoch_start_time = time.time()
        self.model.train()
        loss_total = 0
        cnt = 0
        train_loader.shuffle()
        for i, (inputs, target) in enumerate(train_loader.get_iterator()):
            inputs = torch.Tensor(inputs).to(args.get("device")).transpose(1, 3)
            target = torch.Tensor(target).to(args.get("device")).transpose(1, 3)
            inputs = F.pad(inputs, (1, 0, 0, 0))
            self.model.zero_grad()
            forecast = self.model(inputs).transpose(1, 3)
            forecast = torch.unsqueeze(forecast[:, 0, :, :], dim=1)
            target = torch.unsqueeze(target[:, 0, :, :], dim=1)
            loss = criterion(forecast, target)
            cnt += 1
            loss.backward()
            optimizer.step()
            loss_total += float(loss)

        # print('Epoch {:2d} | Time: {:4.2f}s | Total Loss: {:5.4f}'.format(epoch + 1, (
        #         time.time() - epoch_start_time), loss_total))
        self.save_model(result_file, epoch)
        if (epoch + 1) % args.get("exponential_decay_step") == 0:
            lr_scheduler.step()
        if (epoch + 1) % args.get("validate_freq") == 0:
            is_best = False
            # print('------ VALIDATE ------')
            performance_metrics = \
                self.validate_model(valid_loader, args.get("device"), args.get("norm_method"), args.get("horizon"), scaler=scaler)
            valid_frame = valid_frame.append({"epoch" : epoch, **performance_metrics}, ignore_index=True)
            if np.abs(best_validate_mae) > np.abs(performance_metrics['mae']):
                best_validate_mae = performance_metrics['mae']
                is_best = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            if is_best:
                self.save_model(result_file)
        if args.get("early_stop") and validate_score_non_decrease_count >= args.get("early_stop_step"):
            break
    return dict(valid=valid_frame)