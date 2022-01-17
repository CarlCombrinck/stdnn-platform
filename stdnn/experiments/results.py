import numpy as np
import pandas as pd
import os
import sys

# TODO Deep copies

class Result():
    """
    Class for storing and managing the results of an experiment
    """

    def __init__(self, results={}):
        self.results = {}
        for key, frame in results.items():
            self.add_dataframe(frame, key)

    def add_dataframe(self, dataframe, key):
        self.results[key] = dataframe.copy(deep=True)

    def get_dataframe(self, key):
        return self.results.get(key)

    def get_dataframes(self):
        return self.results

    def get_dataframe_names(self):
        return self.results.keys()

    def copy(self):
        return Result(self.results)

    def __str__ (self):
        return str(self.results)

    def __repr__(self):
        return "Result(\n" + str(self) + "\n)"

class RunResult(Result):
    """
    Class representing the results of an experiment run
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return RunResult(self.results)

class ExperimentResult(Result):
    """
    Class representing the results of an experiment (combined runs)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return ExperimentResult(self.results)