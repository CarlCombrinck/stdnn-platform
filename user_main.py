from user_models import GraphWaveNet
from user_model_managers import GWNManager
from user_plotter import CustomGWNPlotter
from user_preprocessing.utils import (
    process_adjacency_matrix,
    get_node_count_from_data
)

from stdnn.experiments.experiment import ExperimentManager, ExperimentConfigManager
from stdnn.experiments.results import ExperimentResultSet

import argparse
import os
import warnings
from stdnn.reporting.latex import ToLatex

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import torch


def str2bool(v):
    """
    Converts a string argument to the boolean equivalent

    Parameters
    ----------
    v : Union[bool, str]
        Command line argument value

    Returns
    -------
    bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparse_setup():
    """
    Setup command line argument parser

    Returns
    -------
    argparse.Namespace
        A Namespace containing all the command line argument values
    """

    warnings.filterwarnings("ignore", category=UserWarning)

    # TODO Organize parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_pipeline', type=str2bool, default=True)
    parser.add_argument('--model', type=str, default='StemGNN')
    parser.add_argument('--baseline', type=str2bool, default=False)
    parser.add_argument('--baseline_only', type=str2bool, default=False)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--evaluate', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='JSE_clean_truncated')
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--train_length', type=float, default=6)
    parser.add_argument('--valid_length', type=float, default=2)
    parser.add_argument('--test_length', type=float, default=2)
    parser.add_argument('--multi_layer', type=int, default=5)
    parser.add_argument('--norm_method', type=str, default='z_score')
    parser.add_argument('--optimizer', type=str, default='RMSProp')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--early_stop', type=str2bool, default=False)
    # Missing early stop step argument?
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)

    # GWN arguments
    parser.add_argument('--adj_data', type=str2bool, default=False)
    parser.add_argument('--adj_type', type=str, default='double_transition')
    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--gcn_bool', type=str2bool, default=True)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--apt_only', type=str2bool, default=True)
    parser.add_argument('--adapt_adj', type=str2bool, default=True)
    parser.add_argument('--random_adj', type=str2bool, default=True)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    # LSTM arguments
    parser.add_argument('--lstm_layers', type=int, default=100)
    parser.add_argument('--lstm_node', type=int, default=0)

    # User argument configuration
    args = parser.parse_args()

    if args.adj_data:
        adj_matrix = process_adjacency_matrix(os.path.join(
            'data', args.dataset + '.csv'), args.adj_type)
        args.supports = [torch.tensor(i).to(args.device) for i in adj_matrix]
        if args.apt_only:
            args.supports = None
            args.adj_init = None
        else:
            if args.random_adj:
                args.adj_init = None
            else:
                args.adj_init = args.supports[0]
    else:
        args.adj_matrix = None
        args.adj_init = None
        args.supports = None

    args.node_cnt = get_node_count_from_data(args.dataset)

    return args


def create_output_dirs(args):
    result_train_file = os.path.join('output', args.model, args.dataset, str(
        args.window_size), str(args.horizon), 'train')
    baseline_train_file = os.path.join('output', 'lstm', args.dataset, str(
        args.window_size), str(args.horizon), 'train')
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(baseline_train_file):
        os.makedirs(baseline_train_file)

    return result_train_file


def main():

    args = argparse_setup()

    result_train_file = create_output_dirs(args)

    # Hyper parameter configuration
    cs = CS.ConfigurationSpace(seed=1234)
    lr = CSH.UniformFloatHyperparameter(
        'lr', lower=1e-6, upper=1e-3, log=True, meta={"config": "train"})
    dropout = CSH.UniformFloatHyperparameter(
        'dropout', lower=0.2, upper=0.8, log=False, meta={"config": "model"})
    cs.add_hyperparameters([lr])

    # Pipeline and model configuration
    pipeline_config = {
        "model": {
            "meta": {
                "type": GraphWaveNet,
                "manager": GWNManager
            },
            "params": dict(device=args.device, node_cnt=args.node_cnt, dropout=args.dropout_rate,
                           supports=args.supports, gcn_bool=args.gcn_bool, adapt_adj=args.adapt_adj,
                           adj_init=args.adj_init, in_dim=args.in_dim, out_dim=args.horizon,
                           residual_channels=args.channels, dilation_channels=args.channels,
                           skip_channels=args.channels * 8, end_channels=args.channels * 16
                           )
        },
        "preprocess": {
            "params": dict(
                args=args, datafile=args.dataset, result_file=result_train_file
            )
        },
        "train": {
            "params": dict(
                args=vars(args), result_file=result_train_file
            )
        },
        "test": {
            "params": dict(
                args=vars(args), result_train_file=result_train_file
            )
        }
    }

    # Experiment configuration
    experiment_config = {
        "config_space": cs,
        "grid": dict(
            lr=3
        ),
        "runs": 2
    }

    raw_results = None

    if args.run_pipeline:
        exp_config = ExperimentConfigManager(
            pipeline_config, experiment_config)
        experiment_manager = ExperimentManager(exp_config)

        # Run experiment
        raw_results = experiment_manager.run_experiments()

        # Save results
        raw_results.save_to("results", "results.pickle")

    else:
        # Load results
        raw_results = ExperimentResultSet.load_from(
            "results", "results.pickle")

    # Format results
    training_results = {
        label: result.aggregate(group_by="epoch", which=[
                                "train", "valid"], join=True).get_dataframes()
        for label, result in raw_results.get_results().items()
    }
    adj_matrix_results = {
        label: result.aggregate(group_by="index", which=[
                                "adj"], join=False).get_dataframes()
        for label, result in raw_results.get_results().items()
    }

    # Plot results
    CustomGWNPlotter.plot_lines(
        figure_name="MAPE Loss", 
        x="epoch", 
        y=["mape_mean"], 
        std_error=["mape_std_dev"], 
        dataframes_dict=training_results, 
        marker=".",
        save_dir="plots",
        xlabel="Epoch", 
        ylabel="MAPE", 
        title="Training and Validation Loss", 
        perconfig=True
    )
    
    CustomGWNPlotter.plot_adaptive_adj_matrix(
        figure_name='GWN Adaptive Adjacency Matrix',
        dataframe=adj_matrix_results,
        save_dir="plots"
    )

    # Outputting to Latex

    ToLatex.to_latex(training_results)
    ToLatex.to_latex(adj_matrix_results)


if __name__ == '__main__':
    main()
