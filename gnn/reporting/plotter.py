from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys


class Plotter:
    """
    Plotter is a class with static methods that allow a user to create plots and have these plots be saved to the reporting/images directory
    """
    absolutepath = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absolutepath)
    # Navigate to Images directory
    newPath = os.path.join(fileDirectory, 'images')

    @staticmethod
    def plot_correlation_matrix(dataframe, seaborn_theme='white', figure_name="correlation_matrix", save_figure_format='png', cmap_diverging_palette_husl_colours=[150, 275]):
        '''
        Create a correlation matric for a given Pandas Dataframe. 

            Parameters: 
                    dataframe(pandas.dataframe): A pandas dataframe containing the raw data
                    seaborn_theme (string): Input for the seaborn.set_theme() method 
                    figure_name (string): The filename for the figure created by this method
                    save_figure_format (string): the file format (PNG, JPEG, etc) for the plot that will be saved to an external directory
                    cmap_diverging_palette_husl_colours (float[]): a float array containing the HUSL colours [h_neg, h_pos] for the extents of the heatmap of the correlation matrix

            Returns:
                    Void
        '''
        sns.set_theme(style=seaborn_theme)
        corr_matrix = dataframe.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.savefig(newPath + "\\"+figure_name + "." + save_figure_format)


# Generate a large random dataset - just here for testing purposes
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))


Plotter.plot_correlation_matrix(dataframe=d)
