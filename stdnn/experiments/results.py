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

class ResultSet():
    """
    Class for storing and managing many Result objects
    """

    def __init__(self):
        self.results = {}
        self.count = 0

    def add_result(self, result, key=None):
        self.count += 1
        self.results[key if key is not None else str(self.count)] = result.copy()

    def add_results(self, results):
        if isinstance(results, dict):
            for key, result in results.items():
                self.add_result(result, key)
        elif isinstance(results, list):
            for result in results:
                self.add_result(result)
        else:
            raise TypeError(f"Expecting list or dict, got {type(results)}")

    def get_result(self, key):
        return self.results.get(key)

    def get_results(self):
        return self.results

    def __str__(self):
        return str(self.results)

    def __repr__(self):
        return "ResultSet(\n" + str(self) + "\n)"

class RunResultSet(ResultSet):
    """
    Class for storing and managing many RunResult objects
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine(self):
        combined = ExperimentResult()
        temp = {}
        for exp_key, result in self.results.items():
            for frame_key in result.get_dataframe_names():
                if frame_key in temp:
                    temp.get(frame_key).get("frames").append(result.get_dataframe(frame_key))
                    temp.get(frame_key).get("labels").append(exp_key)
                else:
                    temp[frame_key] = {
                        "frames" : [result.get_dataframe(frame_key)],
                        "labels" : [exp_key]
                    }
        for frame_key, data in temp.items():
            combined.add_dataframe(
                pd.concat(
                    data.get("frames"), keys=data.get("labels")
                ), frame_key
            )
        return combined

    # TODO Generalize to apply any aggregation function(s) across each dataframe 
    # TODO Choice of which frames to aggregate?
    def aggregate(self, group_by, columns=None):
        combined = self.combine()
        aggregated = ExperimentResult()
        for key, frame in combined.get_dataframes().items():
            grouped = frame.groupby(by=group_by)
            if columns is not None:
                grouped = grouped[columns]
            means = grouped.mean(numeric_only=True)
            means.columns = [(name + "_mean") for name in means.columns if name != group_by]
            devs = grouped.std()
            devs.columns = [(name + "_std_dev") for name in devs.columns if name != group_by]
            aggregated.add_dataframe(means.join(devs, on=group_by).reset_index(), key)
        return aggregated

class ExperimentResultSet(ResultSet):
    """
    Class for storing and managing many ExperimentResult objects
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)