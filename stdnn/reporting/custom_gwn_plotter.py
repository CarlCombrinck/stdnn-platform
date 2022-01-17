from stdnn.reporting.plotter import Plotter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


class CustomGWNPlotter(Plotter):

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
        plt.savefig(f"{figure_name}.{save_figure_format}")
        plt.clf()

    def plot_network(dataframe, n = 5):
        corr = dataframe.corr()
        v_corr = corr.values
        graph = nx.Graph()
        edges = {}

        for i, a in enumerate(v_corr):
            idx = np.argpartition(np.delete(a, i), -n)[-n:]
            edges[corr.columns[i]] = \
                np.delete(corr.columns[idx].values, np.where(corr.columns[idx].values == corr.columns[i]))

        for k, v in edges.items():
            for n_ in v:
                graph.add_edge(k, n_)
                for e in edges[n_]:
                    if graph.has_edge(k, e) or k == e:
                        continue
                    graph.add_edge(k, e)
        
        nx.draw(graph)
