from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)
# Navigate to Images directory
newPath = os.path.join(fileDirectory, 'images')


class Plotter:

    @staticmethod
    def plot_correlation_matrix(dataframe, seaborn_theme='white', figure_name="correlation_matrix", save_figure_format='png', cmap_diverging_palette_husl_colours=[150, 275]):
        sns.set_theme(style=seaborn_theme)
        corr_matrix = dataframe.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(
            cmap_diverging_palette_husl_colours[0], cmap_diverging_palette_husl_colours[1], as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.savefig(newPath + "\\"+figure_name + "." + save_figure_format)


# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))


Plotter.plot_correlation_matrix(dataframe=d)
