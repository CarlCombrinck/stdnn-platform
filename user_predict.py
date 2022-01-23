import torch
import numpy as np

def predict(self, data_loader, device='cpu'):
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