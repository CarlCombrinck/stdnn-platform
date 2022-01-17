from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
import os
import sys


class Plotter:
    """
    Plotter is a class with static methods that allow a user to create plots and have these plots be saved to the reporting/images directory
    """

    @staticmethod
    def plot_correlation_matrix(dataframe, seaborn_theme='white', figure_name="correlation_matrix", save_figure_format='png', cmap_diverging_palette_husl_colours=[150, 275]):
        """[summary]

        Parameters
        ----------
        dataframe : pandas.dataframe
            A pandas dataframe containing the raw data
        seaborn_theme : str, optional
            Input for the seaborn.set_theme() method, by default 'white'
        figure_name : str, optional
            The filename for the figure created by this method, by default "correlation_matrix"
        save_figure_format : str, optional
            the file format (PNG, JPEG, etc) for the plot that will be saved to an external directory, by default 'png'
        cmap_diverging_palette_husl_colours : list, optional
            a float array containing the HUSL colours [h_neg, h_pos] for the extents of the heatmap of the correlation matrix, by default [240, 10]
        """
        sns.set_theme(style=seaborn_theme)
        corr_matrix = dataframe.corr()
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, cmap=cmap, center=0).set(title="Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f"{figure_name}.{save_figure_format}")
        plt.clf()

    @staticmethod
    def plot_lines(figure_name, x, y, dataframes_dict , std_error = None, save_figure_format='png', **kwargs):
        """[summary]

        Parameters
        ----------
        figure_name : string
            The filename for the figure created by this method
        x : string
            The column name for the x column in the dataframe  
        y : string[]
            An array of strings of the y-axis column names for each dataframe 
        dataframes_dict : dictionary 
            Dictionary with key-value pairs for each config: dataframe 
        save_figure_format : str, optional
            The file format (PNG, JPEG, etc) for the plot that will be saved to an external directory, by default 'png'
        """
        for name, frame in dataframes_dict.items():
            if len(std_error) is None:
                for y_value in y:
                    plt.plot(x, y_value, data = frame, label = f"{name}-{y_value}")
            elif len(y) == len(std_error):
                for y_value, y_std_dev in zip(y, std_error):
                    plt.errorbar(x, y_value, yerr=y_std_dev, data=frame, label=name, **kwargs)
        plt.title(f"{', '.join(y)} vs {x}")
        plt.xlabel(x)
        plt.ylabel(", ".join(y))
        plt.legend(loc="upper right", title="key")
        plt.tight_layout()
        plt.savefig(f"{figure_name}.{save_figure_format}")
        plt.clf()