import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg

def transform_(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data


def inverse_transform_(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


def correlation_adjacency_matrix(dataset):
    return pd.read_csv(dataset).corr().to_numpy()


def symmetric_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asymmetric_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    identity = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - identity
    return L.astype(np.float32).todense()


def process_data(data, window_size, horizon):
    """
    Transforms a two-dimensional input (N x T) into a four-dimensional dataset,
    where N is the number of nodes and T is the steps.

    Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. 2018.
    Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.

    Parameters
    ----------
    data : numpy.ndarray
        Input dataset
    window_size : int
        Input sequence length
    horizon : int
        Output sequence length

    Returns
    -------
    numpy.ndarray
    """
    x_offsets = np.sort(np.concatenate((np.arange(-(window_size - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (horizon + 1), 1))
    samples, nodes = data.shape[0], data.shape[1]
    data = np.expand_dims(data, axis=-1)
    data = np.concatenate([data], axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def process_adjacency_matrix(adj_data, adj_type):
    """
    Preprocesses a Graph WaveNet adjacency matrix

    Parameters
    ----------
    adj_data : str
        File containing adjacency matrix data
    adj_type : str
        Adjacency matrix transformation type

    Returns
    -------
    [numpy.ndarray]
    """
    adj = correlation_adjacency_matrix(adj_data)
    if adj_type == "scaled_laplacian":
        adj = [calculate_scaled_laplacian(adj)]
    elif adj_type == "normalized_laplacian":
        adj = [calculate_normalized_laplacian(adj).astype(np.float32).todense()]
    elif adj_type == "symmetric_adjacency" or adj_type == "transition":
        adj = [symmetric_adjacency(adj)]
    elif adj_type == "double_transition":
        adj = [asymmetric_adjacency(adj), asymmetric_adjacency(np.transpose(adj))]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error
    return adj
