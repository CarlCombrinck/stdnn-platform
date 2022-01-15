class ExperimentConfig():
    """
    Class to represent and manage the parameters for running
    an experiment (a single pass through the ML pipeline)
    """

    def __init__(self, config):
        self.config = config
    
    @property
    def model_type(self):
        return self.config.get("model").get("meta").get("type")

    @property
    def model_manager(self):
        return self.config.get("model").get("meta").get("manager")

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
    def __init__(self, config):
        self.raw_config = config

    # TODO Refactor to yield next experiment configuration
    def next_configuration(self):
        return ExperimentConfig(self.raw_config)

class Experiment():
    """
    Class representing a configured ML pipeline, responsible for executing this pipeline
    with the specified parameters and producing results
    """
    def __init__(self, config):
        self.config = config
        self.results = None

    # TODO Refactor to use results class/add explicit validation?
    def run(self):
        model = self.config.model_type(**self.config.get_model_params())
        model_manager = self.config.model_manager()
        model_manager.set_model(model)
        train_results = model_manager.train_model(**self.config.get_training_params())
        test_results = model_manager.test_model(**self.config.get_testing_params())
        self.results = (train_results, test_results)

    def get_results(self):
        return self.results

class ExperimentManager():
    """
    Class for managing the running of all experiments and collation of results
    """
    def __init__(self, config):
        self.config = config

    # TODO Add loop over configurations and collate results
    # TODO Rerun experiments and aggregate results
    def run_experiments(self):
        params = self.config.next_configuration()
        experiment = Experiment(params)
        experiment.run()
        # TODO Add to result set
        return experiment.get_results()