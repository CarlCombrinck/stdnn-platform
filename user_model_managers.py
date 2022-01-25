from stdnn.experiments.results import RunResult
from stdnn.models.manager import STModelManager

class GWNManager(STModelManager):

    def __init__(self):
        super().__init__()

    def run_pipeline(self, config):
        """
        Executes the machine learning pipeline for the given model

        Parameters
        ----------
        config : ExperimentConfig
            An ExperimentConfig object containing the parameters for the model and pipeline

        Returns
        -------
        results : RunResult
            A RunResult containing the results collected in the pipeline
        """
        train, valid, test, train_scaler, test_scaler = self.preprocess(**config.get_preprocessing_params())
        train_results = self.train_model(train, valid, train_scaler, **config.get_training_params())
        test_results = self.test_model(test, test_scaler, **config.get_testing_params())
        result = RunResult(
            {**train_results, **test_results}    
        )
        return result

    from user_preprocess import preprocess

    from user_train import train_model

    from user_validate import validate_model

    from user_test import test_model

    from user_predict import predict as _custom_inference


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

#         train_set = user_preprocessing.loader.ForecastDataset(train_data, window_size=args.window_size,
#                                                             horizon=args.horizon, normalize_method=args.norm_method,
#                                                             norm_statistic=norm_statistic)
#         valid_set = user_preprocessing.loader.ForecastDataset(valid_data, window_size=args.window_size,
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
#         test_set = user_preprocessing.loader.ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
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


