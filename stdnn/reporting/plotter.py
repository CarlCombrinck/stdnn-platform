from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
import os
import sys


absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# Navigate to Images directory
newPath = os.path.join(fileDirectory, 'images')


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
            a float array containing the HUSL colours [h_neg, h_pos] for the extents of the heatmap of the correlation matrix, by default [150, 275]
        """
        sns.set_theme(style=seaborn_theme, )
        corr_matrix = dataframe.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title="Correlation Matrix")
        plt.savefig(newPath + "\\"+figure_name + "." + save_figure_format)
        plt.clf()

    @staticmethod
    def plot_training_vs_validaion_loss(dataframe, figure_name="training_vs_validation_loss", save_figure_format='png'):
        """[summary]

        Parameters
        ----------
        dataframe : [type]
            [description]
        figure_name : str, optional
            [description], by default "training_vs_validation_loss"
        save_figure_format : str, optional
            [description], by default 'png'
        """
        """dataframe.plot.line(x='epochs', y=['training_loss', 'validation_loss'])
        plt.minorticks_on()
        plt.xticks(list(plt.xticks()[0]) + [0.1])
        plt.setp(visible = True)
        plt.tick_params(axis='both', which='major', direction='out')
        plt.title("Training vs Validation loss by number of epochs")
        plt.ylabel("Accuracy")
        plt.xlabel("# of Epochs")
        plt.legend(['training_loss', 'validation_loss'], loc="upper right")"""
        fig, ax = plt.subplots()
        sns.lineplot(x='epochs', y='Accuracy', hue='Key',
                     data=pd.melt(dataframe, 'epochs', value_name="Accuracy", var_name="Key"), legend= "auto")

        plt.savefig(newPath + "\\"+figure_name + "." + save_figure_format)
        plt.clf()


# Generate a large random dataset - just here for testing purposes
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(50, 26)),
                 columns=list(ascii_letters[26:]))

rs = np.random.RandomState(20)
colnames = ['training_loss', 'validation_loss']
df = pd.DataFrame(data=rs.normal(size=(50, 2)),
                  columns=colnames)
rng = np.arange(start=1, stop=51)
df['epochs'] = rng

Plotter.plot_correlation_matrix(dataframe=d)
Plotter.plot_training_vs_validaion_loss(dataframe=df)
