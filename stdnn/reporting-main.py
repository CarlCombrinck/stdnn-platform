from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
import os
import sys

from experiments.results import Results
from reporting.custom_gwn_plotter import CustomGWNPlotter

if __name__ == '__main__':
    df = pd.read_csv("stdnn\data\JSE_clean_truncated.csv")
    df_s = pd.read_csv("stdnn\data\GWN_corr.csv")

    # Filepath stuff
    absolutepath = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absolutepath)
    # print(fileDirectory)
    # Navigate to Images directory
    newpath = os.path.join(fileDirectory, 'reporting\images')

    df = pd.read_csv("stdnn\data\JSE_clean_truncated.csv")
    df_ = pd.read_csv("stdnn\data\GWN_corr.csv")

    CustomGWNPlotter.plot_correlation_matrix(
        dataframe=df, filepath=newpath+"\\")
    CustomGWNPlotter.plot_adaptive_adj_matrix(
        dataframe=df_, filepath=newpath+"\\")
