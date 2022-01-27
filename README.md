# STDNN-Platform

*An end-to-end, experiment-oriented machine learning pipeline for spatial-temporal data*

## Installation

```
make install
```
or
```
python3 -m venv venv
source ./venv/bin/activate
pip3 install -Ur requirements.txt
```

## Testing GWN Model
```
make test
```
or 
```
source ./venv/bin/activate
python3 user_main.py --model GWN --window_size 40 --horizon 10
```

## Using the Framework

1. Define a PyTorch model class.

```python
import torch as nn

class MyModel(nn.Module):
    ...
```

2. Define a model manager class. \
Methods like `preprocess`, `train_model`, and `test_model` must be implemented with the option of overriding the default `run_pipeline` method. \
These should make use of the model instance stored under `self.model`.

```python
from stdnn.models.manager import STModelManager

class MyModelManager(STModelManager):
    def __init__(self):
        super().__init__()

    def preprocess(self, ...):
        ...
    
    def train_model(self, ...):
        ...

    def test_model(self, ...):
        ...
```

3. Configure the hyperparameters.\
These should make use of the `ConfigSpace` package.\
Each hyperparameter name must match a parameter specified in the pipeline/model configuration. \
The `meta` argument of each hyperparameter must by set to specify where (in what stage of the pipeline) the hyperparameter must be used (model, preprocess, train, test).

```python
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# Hyper parameter configuration
cs = CS.ConfigurationSpace()
my_hyperparameter = CSH.UniformFloatHyperparameter(
    '...', lower=..., upper=..., meta={"config": "..."}
)
my_hyperparameter_2 = CSH.UniformFloatHyperparameter(
    '...', lower=..., upper=..., meta={"config": "..."}
)
# Here two float hyperparameters are added to the config space
cs.add_hyperparameters([my_hyperparameter, my_hyperparameter_2])
```

4. Configure the pipeline and model. \
The parameter dictionaries will be unpacked into the corresponding pipeline method at runtime.\
These should also include the previously-specified hyperparameters with default values (these will be replaced by the correct values during each experiment).

```python
# Pipeline and model configuration
pipeline_config = {
    "model": {
        "meta": {
            "type": MyModel,
            "manager": MyModelManager
        },
        "params" : {...} # passed to model constructor
    },
    "preprocess" : {
        "params" : {...} # passed to preprocess method
    },
    "train": {
        "params" : {...} # passed to train_model method
    },
    "test": {
        "params" : {...} # passed to test_model method
    }
}
```

5. Configure the experiments. 

```python
# Experiment configuration
experiment_config = {
    "config_space": cs,
    # specifies how many values of each hyperparameter to try
    "grid": dict( 
        my_hyperparameter=..., 
        my_hyperparameter_2=...
    ),
    # the number of repeat runs of each experiment
    "runs": ...
}
```

6. Run the experiments.

```python
from stdnn.experiments.experiment import ExperimentManager, ExperimentConfigManager

# Pass config dictionaries to manager
config = ExperimentConfigManager(
            pipeline_config, 
            experiment_config
)
experiment_manager = ExperimentManager(config)

# Run experiment
raw_results = experiment_manager.run_experiments()
```

7. Organise and plot the results.

```python
from stdnn.reporting.plotter import Plotter

results = {
    label: result.aggregate(...).get_dataframes()
    for label, result in raw_results.get_results().items()
}

Plotter.plot_lines(..., dataframes_dict=results, ...)
```