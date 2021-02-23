import matplotlib.pyplot as plt
from itertools import cycle
from scipy import optimize
from core.image.frame_attribute import FrameAttribute
import numpy as np
from PIL import Image
import io
import math
from core.plots.plot_base import viNetPlotBase


class LinePlot(viNetPlotBase):

    def __init__(self, plot_data):
        """
        @param plot_data: must include:
        ground_truth: array of ground truth data
        predicted: corresponding predictions
        confidence: corresponding confidence of prediction
        labels: labels of the ground truth and prediction lists
        """
        super().__init__(plot_data)

    def _confidence_correlation_plot(self, title, confidence_vs, x_label, colors=None, best_fit=True,
                                     adjust_conf=False):
        """
        Creates a scatter plot of confidence vs time. The adds lines of best fits for each label
        @param title: title of plot
        @param colors: list of colors, to color each label
        @param best_fit: Added lines of best fit. Fits the function f = a*x^2 + bx + c
        @param adjust_conf: will adjust confidence value for the case where ground truth != predicted. See
        Comments in code for more
        """
        assert len(confidence_vs) == len(self.groundtruth)
        correlation_groups = [[] for _ in range(len(self.labels))]
        correct_conf = [[] for _ in range(len(self.labels))]

        # Split data into each different ground truths.
        for i in range(len(self.confidence)):
            label_index = self.labels.index(self.groundtruth[i])
            correlation_groups[label_index].append(confidence_vs[i])

            # This adjusts the value of the confidence if prediction != ground truth. The idea is when prediction
            # is wrong we make the new confidence (100 - prediction)/(num_labels - 1) but this method results in some
            # confusions plots. It looks better is the (num_labels - 1) was removed but I don't have a reasoning.
            if not adjust_conf or self.groundtruth[i] == self.predicted[i]:
                correct_conf[label_index].append(self.confidence[i])
            else:
                correct_conf[label_index].append((100 - self.confidence[i]) / max(len(self.labels) - 1, 1))

        # Color cycle to color each label.
        if not colors:
            colors = cycle(['cornflowerblue', 'red', 'darkorange', 'green', 'yellow'])
        else:
            colors = cycle(colors)

        # Plot each of the ground truth groups as a scatter plot and add line of best fit
        fig, ax = plt.subplots(figsize=(11, 9))
        for i, c in zip(range(len(self.labels)), colors):
            plt.scatter(correlation_groups[i], correct_conf[i], marker='.', color=c, label=self.labels[i], alpha=.25)
            # Create lines of best fit for the scatter plots. And then plots them on top of the scatter plots
            if best_fit and len(correlation_groups[i]) > 3:
                ps, params_covariance = optimize.curve_fit(self._best_fit_func, correlation_groups[i], correct_conf[i])

                min_point = min(correlation_groups[i])
                max_point = max(correlation_groups[i])
                change = (max_point - min_point) / 100
                plot_range = [min_point + change * i for i in range(100)]

                ys = [self._best_fit_func(j, ps[0], ps[1], ps[2])
                      for j in plot_range]
                plt.plot(plot_range, ys,
                         label=f'Best fit ({self.labels[i]})', color=c)

        plt.xlabel(x_label)
        plt.ylabel('Network Confidence')
        plt.title(title, fontsize=self.title_fontsize)
        plt.legend(loc="best")
        return fig

    def plot_time_correlation(self, title, inspection_times, colors=None, best_fit=True, adjust_conf=False):
        """
        Creates a scatter plot of confidence vs time. The adds lines of best fits for each label
        @param title: title of plot
        @param colors: list of colors, to color each label
        @param best_fit: Added lines of best fit. Fits the function f = a*x^2 + bx + c
        @param adjust_conf: will adjust confidence value for the case where ground truth != predicted. See
        Comment in code for more
        @param inspection_times: list of inspection times as pandas datetime or anything with hour and minute
        """
        assert len(inspection_times) == len(self.groundtruth)
        # Split data into each different ground truths.
        times = []
        for i in range(len(inspection_times)):
            time = inspection_times[i].hour*60 + inspection_times[i].minute
            times.append(time)

        x_label = 'Time of day (Minutes)'

        return self._confidence_correlation_plot(title, times, x_label, colors, best_fit, adjust_conf)

    def plot_frame_attribute_correlation(self, title, images, fa_func, colors=None, best_fit=True, fa_option=None):
        assert len(images) == len(self.groundtruth)

        fa_results = []
        for i in range(len(images)):
            img = np.array(Image.open(io.BytesIO(images[i])))[..., :3]
            fa = FrameAttribute(img)
            fa_result = fa.attribute_to_single_val(fa_func, fa_option)
            fa_results.append(fa_result)

        x_label = fa_func.replace("_", " ")
        if fa_func == FrameAttribute.dominant_colors.__name__ or fa_func == FrameAttribute.average_color.__name__:
            x_label += f" ({fa_option}ness)"

        return self._confidence_correlation_plot(title, fa_results, x_label, colors, best_fit, False)

    @staticmethod
    def _best_fit_func(x, a, b, c):
        """
        This is the function that the lines of best fit are fitted to. we assume the data will be
        parabolic.
        """
        return a * x * x + b * x + c
