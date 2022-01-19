from stdnn.reporting.plotter import Plotter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os

class CustomGWNPlotter(Plotter):
    """
    Static class for Custom Plotter for GraphWaveNet data. This class allows the user to use existing methods (or add their own) to plot various GWN relevant figures

    Parameters
    ----------
    Plotter : 
        The parent class of this Custom Plotter class
    """

    @staticmethod
    def plot_adaptive_adj_matrix(figure_name, dataframe, save_dir=None, save_figure_format="png", seaborn_theme='white', cmap_diverging_palette_husl_colours=[150, 275]):
        """
        Plots adaptive adjacency matrix data (saved as separate images)

        Parameters
        ----------
        figure_name : str
            The name of the figure
        dataframe : dict
            Dictionary of dataframes (matrices) to plot
        save_figure_format : str, optional
            Image format to save as, by default "png"
        seaborn_theme : str, optional
            Seaborn theme for plotting, by default 'white'
        cmap_diverging_palette_husl_colours : list, optional
            Color palette for matrix cells, by default [150, 275]
        """
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        sns.set_theme(style=seaborn_theme)
        sns.set(font_scale=0.5)
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        fig, ax = plt.subplots()
        row_value = 0
        col_value = 0
        for config, dataframe_dict in dataframe.items():
            for type_of_data, data in dataframe_dict.items():
                columns = data.columns
                data.index = columns.values
                sns.heatmap(data, cmap=cmap, center=0,
                            square=True, linewidths=.5)
                plot_title = f"{figure_name}-{config}-{type_of_data}".replace(
                    ".", "pt")
                plt.title(plot_title)
                plt.tight_layout()
                col_value += 1
                plt.savefig(
                    os.path.join(save_dir, plot_title)
                )
                plt.clf()
