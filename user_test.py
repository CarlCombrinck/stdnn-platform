import user_preprocessing
from user_preprocessing.utils import process_data

# TODO Move to another module (for decorators)
from stdnn.models.utils import timed

import pandas as pd

# @timed(operation_name="Evaluation")
def test_model(self, test_loader, scaler, args, result_train_file):
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

    test_frame = pd.DataFrame(columns=["mae", "mape", "rmse"])
    performance_metrics = self.validate_model(test_loader, args.get("device"), args.get("norm_method"), args.get("horizon"), scaler=scaler)
    test_frame = test_frame.append(performance_metrics, ignore_index=True)
    results["test"] = test_frame
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    # print('Test Set Performance: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.2f}'.format(mape * 100, mae, rmse))
    return results