from user_metrics import evaluate

import torch
import numpy as np

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