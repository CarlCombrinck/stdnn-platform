from stdnn.experiments.results import (
    RunResult, 
    RunResultSet,
    ExperimentResultSet
)
from stdnn.experiments.utils import dictionary_update_deep
from ConfigSpace.util import generate_grid

class ExperimentConfig():
    """
    Class to represent and manage the parameters for running
    an experiment (a single pass through the ML pipeline)
    """

    def __init__(self, config, label):
        """
        Constructor for ExperimentConfig

        Parameters
        ----------
        config : dict
            Dictionary of experiment parameters
        label : str
            A label/name for identifying the experiment configuration
        """
        self.config = dict(config)
        self.label = label
    
    @property
    def model_type(self):
        """
        The type of the model to be configured and passed through the pipeline
        """
        return self.config.get("model").get("meta").get("type")

    @property
    def model_manager(self):
        """
        The type of the model manager to manage the configured model
        """
        return self.config.get("model").get("meta").get("manager")

    def get_label(self):
        """
        Experiment label getter

        Returns
        -------
        str
            The experiment label
        """
        return self.label

    def get_model_params(self):
        """
        Getter for model parameters (shallow copy)

        Returns
        -------
        dict
            A dictionary of model parameters 
            (to be passed to constructor)
        """
        return dict(self.config.get("model").get("params"))

    def get_training_params(self):
        """
        Getter for training parameters (shallow copy)

        Returns
        -------
        dict
            A dictionary of training parameters 
            (to be passed to train method)
        """
        return dict(self.config.get("train").get("params"))

    def get_validation_params(self):
        """
        Getter for validation parameters (shallow copy)

        Returns
        -------
        dict
            A dictionary of validation parameters 
            (to be passed to validate method)
        """
        return dict(self.config.get("validate").get("params"))

    def get_testing_params(self):
        return self.config.get("test").get("params")


class ExperimentConfigManager():
    """
    Class for managing the configuration of the experiments to be run
    """

    def __init__(self, raw_pipeline_config, raw_exp_config):
        """
        Constructor for ExperimentConfigManager

        Parameters
        ----------
        raw_pipeline_config : dict
            Dictionary of structured config data for ML pipeline
        raw_exp_config : dict
            Dictionary of structured config data for the experiments
        """
        self.raw_pipeline_config = dict(raw_pipeline_config)
        self.raw_exp_config = dict(raw_exp_config)
        self.config_space = self.raw_exp_config.get("config_space")
        self._generate_grid()

    def _generate_grid(self):
        """
        Internal method for generating hyperparameter grid
        """
        grid_dims = self.raw_exp_config.get("grid")
        self.grid = generate_grid(self.config_space, grid_dims)

    def get_runs(self):
        """
        Returns the number of repeat runs of the experiment

        Returns
        -------
        int
            The number of repeat runs
        """
        return self.raw_exp_config.get("runs")

    def configurations(self):
        """
        Generator for iterating over each unique configuration of the experiment

        Yields
        -------
        ExperimentConfig
            The next configuration of the experiment
        """
        # Loop over each hyperparameter combination
        for cell in self.grid:
            current_config = dict(self.raw_pipeline_config)
            label = []
            # For each hyperparameter, update (deep) the current configuration with its value
            for param, value in cell.get_dictionary().items():
                key = self.config_space.get_hyperparameter(param).meta.get("config")
                dictionary_update_deep(current_config.get(key), param, value)
                label.append(f"{param}={value}")
            yield ExperimentConfig(current_config, ",".join(label))

class Experiment():
    """
    Class representing a configured ML pipeline, responsible for executing this pipeline
    with the specified parameters and producing results
    """
    def __init__(self, config):
        """
        Constructor for Experiment

        Parameters
        ----------
        config : ExperimentConfig
            The configuration for the experiment
        """
        self.config = config
        self.results = RunResultSet()

    # TODO Refactor to add explicit validation?
    def run(self, repeat=1):
        """
        Run the ML pipeline repeat times with the given configuration

        Parameters
        ----------
        repeat : int, optional
            The number of times to repeat the experiment, by default 1
        """
        for _ in range(repeat):
            model = self.config.model_type(**self.config.get_model_params())
            model_manager = self.config.model_manager()
            model_manager.set_model(model)
            train_results = model_manager.train_model(**self.config.get_training_params())
            test_results = model_manager.test_model(**self.config.get_testing_params())
            result = RunResult(
                {**train_results, **test_results}    
            )
            self.results.add_result(result)

    def get_run_results(self):
        """
        Getter for Run results

        Returns
        -------
        RunResultSet
            The set of run results
        """
        return self.results

class ExperimentManager():
    """
    Class for managing the running of all experiments and collation of results
    """
    def __init__(self, config):
        """
        Constructor for ExperimentManager

        Parameters
        ----------
        config : ExperimentConfigManager
            A config manager for the experiments
        """
        self.config = config

    # TODO Customize aggregation
    # TODO Generalize (too specific in terms of aggregation)
    def run_experiments(self):
        """
        Runs experiment for each configuration and returns
        collated/aggregated results

        Returns
        -------
        ExperimentResultSet
            Set of results for each experiment configuration
        """
        results = ExperimentResultSet()
        for config in self.config.configurations():
            experiment = Experiment(config)
            experiment.run(repeat=self.config.get_runs())
            results.add_result(experiment.get_run_results().combine(), key=config.get_label())
        return results