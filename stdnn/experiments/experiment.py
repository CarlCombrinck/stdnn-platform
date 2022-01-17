from stdnn.experiments.results import (
    RunResult, 
    RunResultSet,
    ExperimentResultSet
)
from ConfigSpace.util import generate_grid

# TODO Deep copies

class ExperimentConfig():
    """
    Class to represent and manage the parameters for running
    an experiment (a single pass through the ML pipeline)
    """

    def __init__(self, config, label):
        self.config = config
        self.label = label
    
    @property
    def model_type(self):
        return self.config.get("model").get("meta").get("type")

    @property
    def model_manager(self):
        return self.config.get("model").get("meta").get("manager")

    def get_label(self):
        return self.label

    def get_model_params(self):
        return self.config.get("model").get("params")

    def get_training_params(self):
        return self.config.get("train").get("params")

    # TODO Implement
    def get_validation_params(self):
        pass

    def get_testing_params(self):
        return self.config.get("test").get("params")


class ExperimentConfigManager():
    """
    Class for managing the configuration of the experiments to be run
    """

    def __init__(self, raw_pipeline_config, raw_exp_config):
        self.raw_pipeline_config = raw_pipeline_config
        self.raw_exp_config = raw_exp_config
        self.config_space = self.raw_exp_config.get("config_space")
        self._generate_grid()

    def _generate_grid(self):
        grid_dims = self.raw_exp_config.get("grid")
        self.grid = generate_grid(self.config_space, grid_dims)

    def get_runs(self):
        return self.raw_exp_config.get("runs")

    # TODO Move to utils?
    @staticmethod
    def _dictionary_update_deep(dictionary, key, value):
        for k, v in dictionary.items():
            if k == key:
                dictionary[key] = value
            elif isinstance(v, dict):
                ExperimentConfigManager._dictionary_update_deep(v, key, value)

    def configurations(self):
        for cell in self.grid:
            current_config = dict(self.raw_pipeline_config)
            label = ""
            for param, value in cell.get_dictionary().items():
                key = self.config_space.get_hyperparameter(param).meta.get("config")
                ExperimentConfigManager._dictionary_update_deep(current_config.get(key), param, value)
                label += f"{param}={value}"
            yield ExperimentConfig(current_config, label)

class Experiment():
    """
    Class representing a configured ML pipeline, responsible for executing this pipeline
    with the specified parameters and producing results
    """
    def __init__(self, config):
        self.config = config
        self.results = RunResultSet()

    # TODO Refactor to use results class/add explicit validation?
    def run(self, repeat=1):
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

    def get_results(self):
        return self.results

class ExperimentManager():
    """
    Class for managing the running of all experiments and collation of results
    """
    def __init__(self, config):
        self.config = config

    # TODO Use result objects
    # TODO Rerun experiments and aggregate results
    # TODO Make abstract?
    def run_experiments(self):
        results = ExperimentResultSet()
        for config in self.config.configurations():
            experiment = Experiment(config)
            experiment.run(repeat=self.config.get_runs())
            results.add_result(experiment.get_results().aggregate(group_by="epoch", which=["valid", "test"]), key=config.get_label())
        return results