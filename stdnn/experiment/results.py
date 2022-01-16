import numpy as np
import pandas as pd
import os
import sys


class Results:
    global results_dict
    results_dict = {}

    def __init__(self):
        return True

    def add_dataframe_to_dictionary(entry_name, dataframe):
        results_dict[entry_name] = dataframe

    def retrieve_dataframe_from_dictionary(entry_name):
        dataframe = results_dict.get(entry_name)
        if dataframe is None:
            return "Cannot retrieve object"
        else:
            return dataframe
