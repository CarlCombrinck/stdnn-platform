import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import stdnn
from stdnn.models.manager import STModelManager
# TODO Remove once refactored
from stdnn.preprocessing.utils import process_data
# TODO Have as property/class method
from stdnn.metrics.error import evaluate
# TODO Move to another module (for decorators)
from stdnn.models.utils import timed
# TODO Remove when moved to plotting
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    def __init__(self, device, node_cnt, dropout=0.3, supports=None, gcn_bool=True, adapt_adj=True, adj_init=None,
                 in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.adapt_adj = adapt_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports
        self.final_adj = None
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and adapt_adj:
            if adj_init is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(node_cnt, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, node_cnt).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(adj_init)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, inputs):
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = inputs
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.adapt_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            if self.supports is not None:
                new_supports = self.supports + [adp]
            else:
                new_supports = [adp]
            self.final_adj = new_supports

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            filter_ = self.filter_convs[i](residual)
            filter_ = torch.tanh(filter_)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter_ * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except BaseException:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.adapt_adj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class GWNManager(STModelManager):

    def __init__(self):
        super().__init__()

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

        scaler = stdnn.preprocessing.loader.CustomStandardScaler(mean=x_train.mean(), std=x_train.std())

        train_loader = stdnn.preprocessing.loader.CustomSimpleDataLoader(scaler.transform(x_train),
                                                                    scaler.transform(y_train), args.get("batch_size"))
        valid_loader = stdnn.preprocessing.loader.CustomSimpleDataLoader(scaler.transform(x_valid),
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

    def _custom_inference(self, data_loader, device='cpu'):
        """
        Performs inference and returns a set of GWN or MTGNN model predictions

        Parameters
        ----------
        model : Union[GraphWaveNet, MTGNN]
            Graph neural network model for inference
        data_loader : Generator
            An iterable data loader
        device : str, optional
            Torch device

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        """
        self.model.eval()
        forecast_set = []
        target_set = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(data_loader.get_iterator()):
                inputs = torch.Tensor(inputs).to(device).transpose(1, 3)
                target = torch.Tensor(target).to(device).transpose(1, 3)[:, 0, :, :]
                forecast_result = self.model(inputs).transpose(1, 3)
                forecast_result = torch.unsqueeze(forecast_result[:, 0, :, :], dim=1)
                forecast_set.append(forecast_result.squeeze())
                target_set.append(target.detach().cpu().numpy())

        return torch.cat(forecast_set, dim=0)[:np.concatenate(target_set, axis=0).shape[0], ...], \
            torch.Tensor(np.concatenate(target_set, axis=0))

    def validate_model(self, data_loader, device, normalize_method, horizon, scaler=None):
        """
        Validates a graph neural network model and returns raw and normalized error metrics
        computed on validation set predictions

        Parameters
        ----------
        model : Union[GraphWaveNet, MTGNN, Model]
            Graph neural network model for validation
        model_name: str,
            Graph neural network model name
        data_loader : torch.Dataset
            An iterable dataset
        device : str
            Torch device
        normalize_method: str
            Raw data normalization method
        statistic: dict
            Raw data statistics
        node_cnt: int
            count of graph nodes
        window_size: int
            Input sequence length or window size
        horizon: int
            Output sequence length or prediction horizon
        scaler: CustomStandardScaler, optional
            Scaler

        Returns
        -------
        dict
        """
        forecast_norm, target_norm = self._custom_inference(data_loader, device)
        mae = ([], [])
        mape = ([], [])
        rmse = ([], [])
        for i in range(horizon):
            if normalize_method:
                if horizon == 1:
                    forecast = torch.Tensor(scaler.inverse_transform(forecast_norm).cpu())
                    target = torch.Tensor(scaler.inverse_transform(torch.squeeze(target_norm)).cpu())
                else:
                    forecast = torch.Tensor(scaler.inverse_transform(forecast_norm[:, :, i]).cpu())
                    target = torch.Tensor(scaler.inverse_transform(target_norm[:, :, i]).cpu())
            else:
                forecast, target = forecast_norm, torch.squeeze(target_norm)

            score = evaluate(target.detach().cpu().numpy(), forecast.detach().cpu().numpy())
            if horizon == 1:
                score_norm = evaluate(torch.squeeze(target_norm).detach().cpu().numpy(),
                                    forecast_norm.detach().cpu().numpy())
            else:
                score_norm = evaluate(target_norm[:, :, i].detach().cpu().numpy(),
                                    forecast_norm[:, :, i].detach().cpu().numpy())

            mape[0].append(score[0])
            mae[0].append(score[1])
            rmse[0].append(score[2])

            mape[1].append(score_norm[0])
            mae[1].append(score_norm[1])
            rmse[1].append(score_norm[2])
            score = (np.mean(mape[0]), np.mean(mae[0]), np.mean(rmse[0]))
            score_norm = (np.mean(mape[1]), np.mean(mae[1]), np.mean(rmse[1]))
        # print("NORM -  MAPE {:>8.4f}% | MAE {:>10.4f} | RMSE {:>10.4f}".format(score_norm[0] * 100, score_norm[1],
        #                                                                     score_norm[2]))
        # print("RAW  -  MAPE {:>8.4f}% | MAE {:>10.4f} | RMSE {:>10.4f}".format(score[0] * 100, score[1], score[2]))
        return dict(mae=score[1], mape=score[0], rmse=score[2])

    # @timed(operation_name="Evaluation")
    def test_model(self, test_data, args, result_train_file):
        """
        Evaluates a GWN or MTGNN model and returns raw and normalized error metrics
        computed on out-of-sample set predictions

        Parameters
        ----------
        test_data : numpy.ndarray
            Test set
        args : argparse.Namespace
            Command line arguments
        result_train_file : str
            Directory to load trained model parameter files
        """
        if not self.has_model():
            self.load_model(result_train_file)

        results = {}

        if self.model.final_adj:
            adj = self.model.final_adj[0].detach().cpu().numpy()
            columns = pd.read_csv('data/' + args.get("dataset") + '.csv').columns
            df = pd.DataFrame(data=adj, columns=columns)
            df.index = columns.values
            df.to_csv(args.get("model") + '_corr.csv')
            results["adj"] = df

        x, y = process_data(test_data, args.get("window_size"), args.get("horizon"))
        scaler = stdnn.preprocessing.loader.CustomStandardScaler(mean=x.mean(), std=x.std())
        test_loader = stdnn.preprocessing.loader.CustomSimpleDataLoader(scaler.transform(x), scaler.transform(y),
                                                                    args.get("batch_size"))
        test_frame = pd.DataFrame(columns=["mae", "mape", "rmse"])
        performance_metrics = self.validate_model(test_loader, args.get("device"), args.get("norm_method"), args.get("horizon"), scaler=scaler)
        test_frame = test_frame.append(performance_metrics, ignore_index=True)
        results["test"] = test_frame
        mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
        # print('Test Set Performance: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.2f}'.format(mape * 100, mae, rmse))
        return results

# class LSTM(nn.Module):
#     """
#     A baseline 100-layer LSTM model with a single output layer for univariate predictions
#     """

#     def __init__(self, input_size=1, hidden_layers=100, output_size=1):
#         """
#         Parameters
#         ----------
#         input_size : int, optional
#             Input layer dimension
#         hidden_layers : int, optional
#             Number of hidden layers
#         output_size : int, optional
#             Output layer dimension
#         """
#         super().__init__()
#         self.hidden_layers = hidden_layers

#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layers)

#         self.linear = nn.Linear(hidden_layers, output_size)

#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layers),
#                             torch.zeros(1, 1, self.hidden_layers))

#     def forward(self, in_seq):
#         lstm_out, self.hidden_cell = self.lstm(in_seq.view(len(in_seq), 1, -1), self.hidden_cell)
#         forecast = self.linear(lstm_out.view(len(in_seq), -1))
#         return forecast


# class LSTMManager(STModelManager):
#     def __init__(self):
#         super().__init__()

#     @timed(operation_name="Train")
#     def train_model(self, train_data, valid_data, args, result_file):
#         """
#         Trains a LSTM model and returns a set of validation performance metrics

#         Parameters
#         ----------
#         train_data : numpy.ndarray
#             Train set
#         valid_data : numpy.ndarray
#             Validation set
#         args : argparse.Namespace
#             Command line arguments
#         result_file : str
#             Directory to store trained model parameter files

#         Returns
#         -------
#         dict
#         """
#         self.model.to(args.device)
#         if len(train_data) == 0:
#             raise Exception('Cannot organize enough training data')
#         if len(valid_data) == 0:
#             raise Exception('Cannot organize enough validation data')

#         if args.norm_method == 'z_score':
#             train_mean = np.mean(train_data[:, args.lstm_node], axis=0)
#             train_std = np.std(train_data[:, args.lstm_node], axis=0)
#             norm_statistic = {"mean": [train_mean], "std": [train_std]}

#         elif args.norm_method == 'min_max':
#             train_min = np.min(train_data[:, args.lstm_node], axis=0)
#             train_max = np.max(train_data[:, args.lstm_node], axis=0)
#             norm_statistic = {"min": [train_min], "max": [train_max]}
#         else:
#             norm_statistic = None
#         if norm_statistic is not None:
#             with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
#                 json.dump(norm_statistic, f)

#         if args.optimizer == 'RMSProp':
#             optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=args.lr)
#         elif args.optimizer == 'SGD':
#             optimizer = torch.optim.SGD(params=self.model.parameters(), lr=args.lr)
#         elif args.optimizer == 'Adagrad':
#             optimizer = torch.optim.Adagrad(params=self.model.parameters(), lr=args.lr)
#         elif args.optimizer == 'Adadelta':
#             optimizer = torch.optim.Adadelta(params=self.model.parameters(), lr=args.lr)
#         else:
#             optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

#         lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)

#         train_set = stdnn.preprocessing.loader.ForecastDataset(train_data, window_size=args.window_size,
#                                                             horizon=args.horizon, normalize_method=args.norm_method,
#                                                             norm_statistic=norm_statistic)
#         valid_set = stdnn.preprocessing.loader.ForecastDataset(valid_data, window_size=args.window_size,
#                                                             horizon=args.horizon, normalize_method=args.norm_method,
#                                                             norm_statistic=norm_statistic)

#         train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
#                                                 num_workers=0)
#         valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

#         criterion = nn.MSELoss(reduction='mean').to(args.device)

#         total_params = 0
#         for name, parameter in self.model.named_parameters():
#             if not parameter.requires_grad:
#                 continue
#             param = parameter.numel()
#             total_params += param
#         print(f"Total Trainable Params: {total_params}")
#         print("LSTM")
#         print()

#         best_validate_mae = np.inf
#         validate_score_non_decrease_count = 0
#         performance_metrics = {}
#         for epoch in range(args.epoch):
#             epoch_start_time = time.time()
#             self.model.train()
#             loss_total = 0
#             cnt = 0
#             for i, (inputs, target) in enumerate(train_loader):
#                 inputs = inputs.to(args.device)
#                 target = target.to(args.device)
#                 optimizer.zero_grad()
#                 self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layers).to(args.device),
#                                     torch.zeros(1, 1, self.model.hidden_layers).to(args.device))
#                 forecast = self.model(inputs[:, :, args.lstm_node])
#                 loss = criterion(forecast, target[:, :, args.lstm_node])
#                 loss.backward()
#                 cnt += 1
#                 optimizer.step()
#                 loss_total += float(loss)
#             print('Epoch {:2d} | Time: {:4.2f}s | Total Loss: {:5.4f}'.format(epoch + 1, (
#                     time.time() - epoch_start_time), loss_total / cnt))
#             self.save_model(result_file, epoch)
#             if (epoch + 1) % args.exponential_decay_step == 0:
#                 lr_scheduler.step()
#             if (epoch + 1) % args.validate_freq == 0:
#                 is_best = False
#                 print('------ VALIDATE ------')
#                 performance_metrics = \
#                     self.validate_model(args.lstm_node, valid_loader, args.device, args.norm_method, norm_statistic)
#                 if args.horizon == 1:
#                     self.validate_model(args.lstm_node, valid_loader, args.device, args.norm_method, norm_statistic,
#                                     True)
#                 if np.abs(best_validate_mae) > np.abs(performance_metrics['mae']):
#                     best_validate_mae = performance_metrics['mae']
#                     is_best = True
#                     validate_score_non_decrease_count = 0
#                 else:
#                     validate_score_non_decrease_count += 1
#                 if is_best:
#                     self.save_model(result_file)
#             if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
#                 break
#         return performance_metrics

#     def validate_model(self, node, data_loader, device, norm_method, statistic, naive=False):
#         """
#         Validates a LSTM or naive model and returns raw and normalized error metrics
#         computed on validation set predictions

#         Parameters
#         ----------
#         model : Union[GraphWaveNet, MTGNN, Model]
#             Graph neural network model for validation
#         node: int
#             Index of node to forecast
#         data_loader : torch.Dataset
#             An iterable dataset
#         device : str
#             Torch device
#         norm_method: str
#             Raw data normalization method
#         statistic: dict
#             Raw data statistics
#         naive: bool
#             Compute last-value (naive) model performance measures

#         Returns
#         -------
#         dict
#         """
#         forecast_set = []
#         target_set = []
#         if naive:
#             for i, (inputs, target) in enumerate(data_loader):
#                 forecast_set.append(inputs[:, -1, node])
#                 target_set.append(target[:, :, node])
#         else:
#             self.model.eval()
#             with torch.no_grad():
#                 for i, (inputs, target) in enumerate(data_loader):
#                     inputs = torch.Tensor(inputs[:, :, node]).to(device)
#                     target_norm = torch.Tensor(target[:, :, node]).to(device)
#                     self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layers),
#                                     torch.zeros(1, 1, self.model.hidden_layers))
#                     forecast_result = self.model(inputs)
#                     forecast_set.append(forecast_result.squeeze())
#                     target_set.append(target_norm.detach().cpu().numpy())

#         forecast_norm = torch.cat(forecast_set, dim=0)[:np.concatenate(target_set, axis=0).shape[0], ...].detach().cpu() \
#             .numpy()
#         target_norm = np.concatenate(target_set, axis=0)
#         if target_norm.shape[1] == 1:
#             target_norm = target_norm[:, 0]

#         if norm_method == 'min_max':
#             scale = statistic['max'] - statistic['min'] + 1e-8
#             forecast = forecast_norm * scale + statistic['min']
#             target = target_norm * scale + statistic['min']
#         elif norm_method == 'z_score':
#             forecast = forecast_norm * statistic['std'] + statistic['mean']
#             target = target_norm * statistic['std'] + statistic['mean']
#         else:
#             forecast, target = forecast_norm, target_norm
#         score = evaluate(target, forecast)
#         score_norm = evaluate(target_norm, forecast_norm)

#         if naive:
#             print("LAST VALUE MODEL")
#         print("NORM -  MAPE {:>8.4f}% | MAE {:>10.4f} | RMSE {:>10.4f}".format(score_norm[0] * 100, score_norm[1],
#                                                                             score_norm[2]))
#         print("RAW  -  MAPE {:>8.4f}% | MAE {:>10.4f} | RMSE {:>10.4f}".format(score[0] * 100, score[1], score[2]))

#         return dict(mae=score[1], mape=score[0], rmse=score[2])

#     @timed(operation_name="Evaluation")
#     def test_model(self, test_data, args, result_train_file):
#         """
#         Evaluates a LSTM model and returns raw and normalized error metrics
#         computed on out-of-sample set predictions

#         Parameters
#         ----------
#         test_data : numpy.ndarray
#             Test set
#         args : argparse.Namespace
#             Command line arguments
#         result_train_file : str
#             Directory to load trained model parameter files
#         """
#         with open(os.path.join(result_train_file, 'norm_stat.json'), 'r') as f:
#             normalize_statistic = json.load(f)
        
#         if not self.has_model():
#             self.load_model(result_train_file)
#         test_set = stdnn.preprocessing.loader.ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
#                                                             normalize_method=args.norm_method)
#         test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
#                                                 shuffle=False, num_workers=0)
#         performance_metrics = self.validate_model(args.lstm_node, test_loader, args.device, args.norm_method,
#                                                 normalize_statistic)
#         mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
#         print('Test Set Performance: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.2f}'.format(mape * 100, mae, rmse))

#         if args.horizon == 1:
#             performance_metrics = self.validate_model(args.lstm_node, test_loader, args.device, args.norm_method,
#                                                     normalize_statistic, True)
#             mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
#             print(
#                 'Last-Value Test Set Performance: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.2f}'.format(mape * 100, mae,
#                                                                                                     rmse))


