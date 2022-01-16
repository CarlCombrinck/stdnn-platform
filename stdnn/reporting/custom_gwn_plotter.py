from string import ascii_letters
from reporting.plotter import Plotter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
import os
import sys


class CustomGNNPlotter(Plotter):

    @staticmethod
    def plot_adaptive_adj_matrix(dataframe, filepath, seaborn_theme='white', figure_name="adaptive_adjacency_matrix", save_figure_format='png', cmap_diverging_palette_husl_colours=[150, 275]):
        """[summary]

        Parameters
        ----------
        dataframe : [type]
            [description]
        filepath : [type]
            [description]
        seaborn_theme : str, optional
            [description], by default 'white'
        figure_name : str, optional
            [description], by default "correlation_matrix"
        save_figure_format : str, optional
            [description], by default 'png'
        cmap_diverging_palette_husl_colours : list, optional
            [description], by default [150, 275]
        """
        sns.set_theme(style=seaborn_theme)
        sns.set(font_scale=0.5)
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        fig, ax = plt.subplots()
        # Hard coded - need to generalise
        columns = pd.read_csv("stdnn\data\JSE_clean_truncated.csv").columns
        dataframe.index = columns.values

        sns.heatmap(dataframe.iloc[::, 1::], cmap=cmap, annot=False, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title="Correlation Matrix")
        plt.tight_layout()
        plt.savefig(filepath+figure_name + "." + save_figure_format)
        plt.clf()
