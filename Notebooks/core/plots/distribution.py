import matplotlib.pyplot as plt
import seaborn as sns
from core.plots.plot_base import viNetPlotBase
import pandas as pd
import math
import numpy as np
import statistics


class Distribution:
    """
        Data distribution and summary plots
        Follow the structure blow (pie chart) when adding a new plot
    """

    def __init__(self, data=None, labels=None, title=None, colors=None):
        self.colors = colors
        self._data = data
        self._labels = labels
        self._title = title
        self.hist = self.Histogram(self._data, self._labels, self._title)
        self.pie = self.Pie(self._data, self._labels, self._title)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exceptions):
        del self._data
        del self._labels

    class Histogram:
        """
        Bar chart, histogram plot
        """
        def __init__(self, data, labels, title):
            """

            @param data:
            @param labels:
            @param title:
            """

            self._data = data
            self._labels = labels
            self._counts = None
            self.title = title

            sns.set_style('whitegrid')

        def plot(self, exceptions=None, xlabel='Category', ylabel='Count', **kwargs):
            plt.rcParams.update({'figure.autolayout': True})
            if isinstance(self._data, dict):
                self._labels = list(self._data.keys())
                self._counts = np.array(list(self._data.values()))
            else:
                assert isinstance(self._labels, list)
                self._counts = self._data
            if not isinstance(self._counts, np.ndarray):
                self._counts = np.array(self._counts)
            if exceptions is None:
                exceptions = ['Unknown']
            try:
                # plt.gcf().subplots_adjust(bottom=0.4)  # give more room for xlabels
                # plt.tight_layout()
                # fig = plt.figure(figsize=(20, 10))  # width:20, height:3
                # ax = fig.add_subplot(111)
                fig, ax = plt.subplots(figsize=(20, 10))
                plt.bar(self._labels, height=self._counts, color='orange')
                plt.xticks(range(len(self._labels)), rotation='vertical')
                total = np.sum(self._counts)
                plt.title(self.title, fontsize=20)
                plt.xlabel(xlabel, fontsize=14)
                plt.ylabel(ylabel, fontsize=14)
                plt.tick_params(labelsize=12)
                i = 0
                colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(self._counts)))
                for p, color in zip(ax.patches, colors):
                    width = p.get_width()
                    height = p.get_height()
                    pct = round(height / total * 100, 2)
                    # set bar color
                    if self._labels[i] in exceptions:
                        p.set_facecolor('r')  # color exception labels with red
                    else:
                        p.set_facecolor(color)
                    x, y = p.get_xy()
                    # annotate each bar with pct% total
                    if pct > 1:
                        ax.annotate(f"{pct}%", (x + width / 2, y + height * 1.02), ha='center')
                    i += 1
                return fig

            except BaseException as e:
                print(e)

        def plot_number_distribution(self, data=None, label='Distance (m)', **kwargs):
            """ Plots the histogram of distribution of values in a list of numbers
            @param label:
            @param data: a dataframe, a series, a list or numpy array
            """
            if not data:
                data = self._data

            assert isinstance(data, (np.ndarray, pd.DataFrame, pd.Series, list))

            if not isinstance(data, list):
                data = list(data)
            bins = 20
            title = 'Distance Distribution, # frames: {}'.format(len(data))

            if 'filter' in kwargs:
                axis_range = tuple(kwargs['filter'])
                axis_min = int(axis_range[0])
                axis_max = int(axis_range[1])
                data = [element for element in data if axis_min < element < axis_max]
            if 'bins' in kwargs:
                bins = int(kwargs['bins'])
            if 'title' in kwargs:
                title = kwargs['title']

            fig, axis = plt.subplots(figsize=(20, 10))
            orient = 'horizontal'
            mean = round(statistics.mean(data))
            maxx = round(max(data))
            mode = round(statistics.mode(data))
            stdev = round(statistics.stdev(data))
            median = round(statistics.median(data))

            if 'orient' in kwargs:
                orient = kwargs['orient']
            if orient == 'horizontal':
                ax = sns.histplot(y=data, bins=bins)
                plt.ylabel(label, fontsize=14)
                plt.xlabel('Count', fontsize=14)
                ax.axhline(mean, color='r', linestyle=':')
                ax.axhline(median, color='r', linestyle='--')
                ax.axhline(mode, color='r', linestyle='-')
                ax.axhline(stdev, color='r', linestyle=None)
                ax.axhline(maxx, color='Orange', linestyle='-.')
            else:
                ax = sns.histplot(x=data, bins=bins)
                plt.xlabel(label, fontsize=14)
                plt.ylabel('Count', fontsize=14)
                ax.axvline(mean, color='r', linestyle=':')
                ax.axvline(median, color='r', linestyle='--')
                ax.axvline(mode, color='r', linestyle='-')
                ax.axvline(maxx, color='Orange', linestyle='-.')
                ax.axvline(stdev, color='r', linestyle=None)

            axis.set_title(title, fontsize=24)
            plt.tick_params(labelsize=16)
            plt.legend({'Mean={}'.format(round(mean)): mean,
                        'Median={}'.format(median): median, 'Mode={}'.format(mode): mode,
                        'Stdev={}'.format(stdev): stdev,
                        'Max={}'.format(maxx): maxx}, loc=0, prop={'size': 12})

            i = 0
            patches = axis.patches
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(patches)))

            for p, color in zip(patches, colors):
                p.set_facecolor(color)
                i += 1

            return fig

        def __dir__(self):
            raise NotImplemented("Not implemented yet!")

    class Pie:
        """
        Pie chart
        """

        def __init__(self, data, labels, title, colors=None):
            """
            TODO - handle various formats, currently this only plots from dict or lists
            TOTO - support for dataframe and pandas series
            
            @param data:
            @param labels:
            @param title:
            @param colors:
            """
            self._title = title
            self._colors = colors
            if (isinstance(data, dict) and labels is None) or \
                    (isinstance(data, list) and isinstance(labels, list)):
                self._data = data
                self._label = labels

        def plot(self, counts=None, labels=None, title=None, colors=None):
            """
            Must supply the arguments when calling this function without
            out object instance

            @param counts:
            @param labels:
            @param title:
            @param colors:
            @return:
            """

            if not counts:
                if isinstance(self._data, dict):
                    counts = self._data.values()
                    labels = self._data.keys()
                else:
                    counts = self._data
                    labels = self._label

                if not title: title = self._title

            m = max(counts)
            max_loc = 0
            for i, j in enumerate(counts):
                if j == m:
                    max_loc = i

            explode = [0 for i in range(len(counts))]
            explode[max_loc] = 0.03
            explode = tuple(explode)

            if colors is None:
                colors = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9', '#C1F0F6', '#0099CC']

            colors = colors[:len(counts)]
            fig, ax = plt.subplots(figsize=(11, 9))

            assert len(labels) > 1
            legends = [x + ", Count: " + str(c) for x, c in zip(labels, counts)]
            if explode:
                _, texts, _ = ax.pie(counts, explode=explode[:len(counts)], labels=None, autopct='%1.1f%%',
                                     shadow=False, startangle=90)
                [text.set_fontsize(24) for text in texts]
            else:
                _, texts, _ = ax.pie(counts, colors=colors, labels=None, autopct='%1.1f%%',
                                     shadow=False, startangle=90)
                [text.set_fontsize(34) for text in texts]

            ax.axis('equal')  # ensure pie is drawn as a circle
            ax.set_title(title, fontsize=20)
            plt.tight_layout()
            plt.legend(legends, loc=0)

            return fig

        @staticmethod
        def donut_plot(outer_group_data, inner_group_data, outer_group_label=None,
                       inner_group_label=None, title='Donut Chart', colors=None):

            assert outer_group_data
            assert inner_group_data
            outer = None
            outer_labels, inner_labels = None, None

            if not outer_group_label:
                assert isinstance(outer_group_data, dict)
                outer_labels = list(outer_group_data.keys())
                outer = list(outer_group_data.values())
            if not inner_group_label:
                assert isinstance(inner_group_data, dict)
                inner_labels = list(inner_group_data.keys())
                inner = list(inner_group_data.values())

            # First Ring (outside)
            fig, ax = plt.subplots()
            ax.axis('equal')
            outerpie, _ = ax.pie(outer, radius=1.3, labels=outer_labels,
                                 colors=colors[:3])
            plt.setp(outerpie, width=0.3, edgecolor='white')

            # Second Ring (Inside)
            innerpie, _ = ax.pie(inner, radius=1.3 - 0.3, labels=inner_labels,
                                 labeldistance=0.7, colors=colors[3:])
            plt.setp(innerpie, width=0.4, edgecolor='white')
            plt.margins(0, 0)
            outer_group_label = ["{} (Count = {})".format(label, count)
                                 for label, count in outer_group_data.items()]
            inner_group_label = ["{} (Count = {})".format(label, count)
                                 for label, count in inner_group_data.items()]
            ax.set_title(title)
            plt.legend(outer_group_label + inner_group_label)

            return fig

    class KDE(viNetPlotBase):

        def __init__(self, plot_data):
            super().__init__(plot_data)
            plt.tight_layout()

        def plot(self, title='KDE plot of network confidence'):
            """
            Kernel Density Estimate plot for confidence of network
            https://en.wikipedia.org/wiki/Kernel_density_estimation
            Not sure why i have to do the two closes but other wise it doesnt work. Also this is a very finicky function
            and might stop working at any moment.
            @param title:
            @return:
            """
            plt.close()
            kde = sns.distplot(list(self.confidence))
            fig = kde.get_figure()
            plt.title(self.pre_title + title, fontsize=self.title_fontsize)
            plt.close()
            return fig

        def plot_multiclass(self, title=""):
            """
            Multi KDE, plots one plot for each label
            """
            sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
            df = pd.DataFrame(dict(Confidence=self.confidence, predicted=self.predicted))

            # Initialize the FacetGrid object
            pal = sns.cubehelix_palette(len(self.labels), rot=-.25, light=.7)
            g = sns.FacetGrid(df, row="predicted", hue="predicted", aspect=10, height=.5, palette=pal)

            # Draw the densities in a few steps
            g.map(sns.kdeplot, "Confidence", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
            g.map(sns.kdeplot, "Confidence", clip_on=False, color="w", lw=2, bw=.2)
            g.map(plt.axhline, y=0, lw=2, clip_on=False)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x, color, label):
                ax = plt.gca()
                ax.text(0, .2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax.transAxes)

            g.map(label, "Confidence")

            # Set the subplots to overlap
            g.fig.subplots_adjust(hspace=-.25)

            # Remove axes details that don't play well with overlap
            g.set_titles("")
            g.set(yticks=[])
            g.despine(bottom=True, left=True)
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle(self.pre_title + title)

            return g

    class ScatterPlot(viNetPlotBase):
        """

        """
        def __init__(self, plot_data, title):
            super().__init__(plot_data)
            self._data = plot_data

        def plot(self, column):
            assert isinstance(self._data, pd.DataFrame)
            self._data[column].plot()
            plt.show()

        def plot_categorical_scatter_plot(self, title="", correct_predictions_only=False, colors=None):
            """

            @param title:
            @param correct_predictions_only:
            @param colors:
            @return:
            """
            plt.close()
            confidence = 'Confidence'
            predicted = 'Predicted'
            groundtruth = 'Groundtruth'

            if colors is None:
                colors = ["r", "c", "y", "g"]

            df = pd.DataFrame({groundtruth: self.groundtruth,
                               predicted: self.predicted,
                               confidence: self.confidence})

            if correct_predictions_only:
                df = df[df[groundtruth] == df[predicted]]

            rows_per_point = math.ceil(df.shape[0] / 1000)

            dfs = []
            for l in self.labels:
                if l not in df[groundtruth].values:
                    continue
                df_gt_class = df.loc[df.groupby(groundtruth).groups[l]]

                for k in self.labels:
                    if k not in df_gt_class[predicted].values:
                        continue
                    gt_pred_df = df_gt_class.loc[df_gt_class.groupby(predicted).groups[k]].sort_values(by=[confidence])
                    dfs.append(gt_pred_df)

            # starting creating new data frame
            new_gt_str = f"{groundtruth} ({rows_per_point} classifications per dot)"
            reduced_df = pd.DataFrame(columns=[new_gt_str, predicted, confidence])
            for sub_df in dfs:
                if sub_df.shape[0] is None or sub_df.shape[0] == 0:
                    continue

                for ind in range(0, sub_df.shape[0], rows_per_point):
                    sub_sec = sub_df[confidence][ind:min(ind + rows_per_point, sub_df.shape[0])]
                    if len(sub_sec) == 0:
                        continue
                    avg_conf = sum(sub_sec) / len(sub_sec)
                    reduced_df = reduced_df.append({new_gt_str: sub_df[groundtruth].iloc[0],
                                                    predicted: sub_df[predicted].iloc[0], confidence: avg_conf},
                                                   ignore_index=True)

            # Draw a categorical scatterplot to show each observation
            fig, ax = plt.subplots(figsize=(11, 9))
            sns.swarmplot(x=predicted, y=confidence, hue=new_gt_str,
                          palette=colors, data=reduced_df, ax=ax)
            plt.title(self.pre_title + title, fontsize=self.title_fontsize)
            plt.close()

            return fig
