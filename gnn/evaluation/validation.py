import os

import numpy as np
import torch
import torch.utils.data

from gnn.metrics.error import evaluate
from gnn.utils import inverse_transform_


def custom_inference(model, data_loader, device='cpu'):
    model.eval()
    forecast_set = []
    target_set = []
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader.get_iterator()):
            inputs = torch.Tensor(inputs).to(device).transpose(1, 3)
            target = torch.Tensor(target).to(device).transpose(1, 3)[:, 0, :, :]
            forecast_result = model(inputs).transpose(1, 3)
            forecast_result = torch.unsqueeze(forecast_result[:, 0, :, :], dim=1)
            forecast_set.append(forecast_result.squeeze())
            target_set.append(target.detach().cpu().numpy())

    return torch.cat(forecast_set, dim=0)[:np.concatenate(target_set, axis=0).shape[0], ...], \
           torch.Tensor(np.concatenate(target_set, axis=0))


def inference(model, data_loader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, _ = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)


def validate(model, model_name, data_loader, device, normalize_method, statistic,
             node_cnt, window_size, horizon, result_file=None, scaler=None):
    if model_name == 'StemGNN':
        forecast_norm, target_norm = inference(model, data_loader, device,
                                               node_cnt, window_size, horizon)
    else:
        forecast_norm, target_norm = custom_inference(model, data_loader, device)
    if model_name == 'StemGNN':
        if normalize_method:
            forecast = inverse_transform_(forecast_norm, normalize_method, statistic)
            target = inverse_transform_(target_norm, normalize_method, statistic)
        else:
            forecast, target = forecast_norm, target_norm
        score = evaluate(target, forecast)
        score_norm = evaluate(target_norm, forecast_norm)
    else:
        mae = ([], [])
        mape = ([], [])
        rmse = ([], [])
        for i in range(horizon):
            if normalize_method:
                if horizon == 1:
                    forecast = torch.Tensor(scaler.inverse_transform(forecast_norm))
                    target = torch.Tensor(scaler.inverse_transform(torch.squeeze(target_norm)))
                else:
                    forecast = torch.Tensor(scaler.inverse_transform(forecast_norm[:, :, i]))
                    target = torch.Tensor(scaler.inverse_transform(target_norm[:, :, i]))
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
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file and model_name == 'StemGNN':
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forecasting_2d = forecast[:, step_to_print, :]
        forecasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forecasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forecasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forecasting_2d - forecasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forecasting_2d - forecasting_2d_target) / forecasting_2d_target), delimiter=",")

    return dict(mae=score[1], mape=score[0], rmse=score[2], )


def validate_baseline(model, data_loader, device, norm_method):
    model.eval()
    forecast_set = []
    target_set = []
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader):
            inputs = torch.Tensor(inputs[:, :, 0]).to(device)
            target_norm = torch.Tensor(target[:, :, 0]).to(device)
            model.hidden = (torch.zeros(1, 1, model.hidden_layers),
                            torch.zeros(1, 1, model.hidden_layers))
            forecast_result = model(inputs)
            forecast_set.append(forecast_result.squeeze())
            target_set.append(target_norm.detach().cpu().numpy())

    forecast_norm = torch.cat(forecast_set, dim=0)[:np.concatenate(target_set, axis=0).shape[0], ...].detach().cpu() \
        .numpy()
    target_norm = np.concatenate(target_set, axis=0)

    if norm_method == 'z_score':
        forecast_scale = np.max(forecast_norm, axis=0) - np.min(forecast_norm, axis=0)
        target_scale = np.max(target_norm, axis=0) - np.min(target_norm, axis=0)
        forecast = forecast_norm * forecast_scale + np.min(forecast_norm, axis=0)
        target = target_norm * target_scale + np.min(target_norm)
    elif norm_method == 'min_max':
        forecast = forecast_norm * np.std(forecast_norm, axis=0) + np.mean(forecast_norm, axis=0)
        target = target_norm * np.std(target_norm, axis=0) + np.mean(target_norm, axis=0)
    else:
        forecast, target = forecast_norm, target_norm
    score = evaluate(target, forecast)
    score_norm = evaluate(target_norm, forecast_norm)

    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')

    return dict(mae=score[1], mape=score[0], rmse=score[2], )
