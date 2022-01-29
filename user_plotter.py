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
    def __determine_number_of_subplots(dict):
        counter = 0
        for config, dataframe_dict in dict.items():
            for type_of_data, data in dataframe_dict.items():
                counter += 1
        return counter

    @staticmethod
    def __determine_number_of_columns(dict):
        counter = 0
        for config, dataframe_dict in dict.items():
            counter += 1
        return counter

    @staticmethod
    def plot_adaptive_adj_matrix(figure_name, dataframe, grouped=False, perconfig=False, save_dir=None, save_figure_format="png", seaborn_theme='white', cmap_diverging_palette_husl_colours=[150, 275]):
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
        plt.clf()
        sns.set_theme(style=seaborn_theme)
        sns.set(font_scale=0.5)
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        if grouped is True:
            values = CustomGWNPlotter.__determine_number_of_subplots(dataframe)
            if values % 2 == 0:
                if values > 4:
                    dims = [4, int(values/4)]
            else:
                dims = [3, int(values % 3)]
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            fig, ax = plt.subplots(nrows=dims[0], ncols=dims[1])
            row_value = 0
            col_value = 0
            for config, dataframe_dict in dataframe.items():
                for type_of_data, data in dataframe_dict.items():
                    columns = data.columns
                    data.index = columns.values
                    plot_title = f"{figure_name}-{config}-{type_of_data}".replace(
                        ".", "pt")
                    sns.heatmap(data, cmap=cmap, center=0, square=True,
                                linewidths=.5, ax=ax[row_value, col_value])
                    ax[row_value, col_value].set_title(plot_title)
                    if(col_value == dims[1]-1):
                        col_value = 0
                        row_value += 1
                    else:
                        col_value += 1
            plt.show()
            plt.savefig(os.path.join(save_dir, figure_name + f": {config}"))
            return

        elif perconfig is True:
            for config, dataframe_dict in dataframe.items():
                plt.clf()
                ncols = CustomGWNPlotter.__determine_number_of_columns(
                    dataframe_dict)
                col_value = 0
                for type_of_data, data in dataframe_dict.items():
                    fig, ax = fig, ax = plt.subplots(nrows=1, ncols=ncols)
                    columns = data.columns
                    data.index = columns.values
                    plot_title = f"{figure_name}-{config}-{type_of_data}".replace(
                        ".", "pt")
                    sns.heatmap(data, cmap=cmap, center=0, square=True,
                                linewidths=.5, ax=ax[1, col_value])
                    ax[1, col_value].set_title(plot_title)
                    col_value += 1
            plt.savefig(os.path.join(save_dir, figure_name))
            return

        else:
            fig, ax = plt.subplots()
            for config, dataframe_dict in dataframe.items():
                for type_of_data, data in dataframe_dict.items():
                    columns = data.columns
                    data.index = columns.values
                    plot_title = f"{figure_name}-{config}-{type_of_data}".replace(
                        ".", "pt")
                    sns.heatmap(data, cmap=cmap, center=0, square=True,
                                linewidths=.5).set(title=plot_title)
                    plt.savefig(os.path.join(save_dir, plot_title))
                    plt.clf()
