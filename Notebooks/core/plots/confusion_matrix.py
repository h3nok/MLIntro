import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from itertools import cycle
from plots.plot_base import viNetPlotBase


class ConfusionMatrix(viNetPlotBase):
    _confusion_matrix = None
    _binary = False

    def __init__(self, plot_data):
        """
        @param plot_data: Must include
        ground_truth: array of ground truth data
        predicted: corresponding predictions
        labels: labels of the ground truth and prediction lists
        confidence: confidence of predictions. Only used for PR curves, and is not needed for CM and CR
        """
        super().__init__(plot_data)

        if len(self.labels) <= 2:
            self._binary = True

    def plot_confusion_matrix(self, title, color=None):
        """
        generates the CM and plots it. Also includes accuracy in the plot
        @param title: Tile for the plot
        @param color: color, in the form of a color map, or string
        @return: fig of the CM plot
        """
        # Create confusion matrix. And find accuracy and tn fp fn tp if binary CM
        self._confusion_matrix = confusion_matrix(self.groundtruth, self.predicted, labels=self.labels)
        acc = accuracy_score(self.groundtruth, self.predicted)

        # Create normalized cm and sum up rows
        cm_normalized = self._confusion_matrix.astype('float') / self._confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_sum = np.sum(self._confusion_matrix, axis=1, keepdims=True)

        if color is None:
            color = plt.get_cmap('Oranges')
        elif isinstance(color, str):
            color = plt.get_cmap(color)

        fig, ax = plt.subplots(figsize=(11, 9))

        # Create wording, or annotation for each of the cells of the CM.
        annots = []
        for i in range(self._confusion_matrix.shape[0]):
            row_sum = cm_sum[i, 0]
            for j in range(self._confusion_matrix.shape[1]):
                annots.append(f"{round(cm_normalized[i, j] * 100, 2)}%\n{self._confusion_matrix[i, j]}/{row_sum}")

        annots = (np.array(annots)).reshape(len(self.labels), len(self.labels))

        # Create the heat map for the CM
        sns.heatmap(self._confusion_matrix, annot=annots, xticklabels=self.labels, yticklabels=self.labels, ax=ax,
                    fmt='', cmap=color, cbar_kws={'label': 'Frame Count'})

        # We will added the accuracy to the bottom of the plot, with x and y labels
        # If we have a binary CM then we will also add the tn fp fn tp, tpr, tnr
        if self._binary:
            if len(self._confusion_matrix) < 2:
                print("Could not ravel Confusion matrix. not enough classes in ground truth. "
                      "Make sure classification group is correct.")
                return None
            tn, fp, fn, tp = self._confusion_matrix.ravel()
            tpr = round((tp / (tp + fn)), 2)
            tnr = round((tn / (tn + fp)), 2)

            ax.set(title=title, ylabel='Groundtruth',
                   xlabel=f'Predicted\nAccuracy={round(acc * 100, 2)}%\ntn={round(tn, 2)}, '
                          f'fp={round(fp, 2)}, fn={round(fn, 2)}, tp={round(tp, 2)}, tpr={tpr}, tnr={tnr}')
        else:
            ax.set(title=title, ylabel='Groundtruth',
                   xlabel=f'Predicted\nAccuracy={round(acc * 100, 2)}%')

        ax.set_title(self.pre_title + title, fontsize=self.title_fontsize)
        fig.tight_layout()

        return fig

    def plot_classification_report(self, title, color=None):
        """
                Creates a table holding the values for the classification report
                @param title: Title for the figure
                @param color: color map
                @return: figure for classification report table
                """
        cr = classification_report(self.groundtruth, self.predicted, labels=self.labels, output_dict=True)
        acc = cr.pop('accuracy', None)

        cr_list = [list(l.values()) for l in list(cr.values())]
        support = [l[-1] for l in cr_list]
        cr_list = [l[:-1] for l in cr_list]

        y_labels = list(cr.keys())
        # y_labels.remove('accuracy')
        x_labels = list(cr[y_labels[0]].keys())[:-1]

        fig, ax = plt.subplots(figsize=(11, 9))

        annotations = []
        # Loop over data dimensions and create text annotations.
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                item = cr.get(y_labels[i], {})
                if not isinstance(item, dict):
                    break
                val = item.get(x_labels[j], "NaN")
                if val == "NaN":
                    annotations.append(val)
                else:
                    annotations.append(f'{round(val, 2)}')

        annots = (np.array(annotations)).reshape(len(y_labels), len(x_labels))

        # Color
        if color is None:
            color = plt.get_cmap('Oranges')
        elif isinstance(color, str):
            color = plt.get_cmap(color)

        # new stuff
        sns.heatmap(cr_list,
                    annot=annots,
                    # annot_kws={"size": 10},
                    xticklabels=[l for l in x_labels if l != 'support'],
                    yticklabels=['{0} ({1})'.format(y_labels[idx], sup) for idx, sup in enumerate(support)],
                    ax=ax, fmt='',
                    cmap=color)

        # We want to show all ticks...
        ax.set(title=title, ylabel='Groundtruth', xlabel='Metric')
        ax.set_title(self.pre_title + title, fontsize=self.title_fontsize)

        fig.tight_layout()

        return fig

    def plot_precision_recall(self, title=None, colors=None):
        """
        Creates multi PR curve plot. If the binary is True then it will create a binary PR curve based on the first
        of the two labels.
        @param title: title for plot
        @param colors: colors for each of the lines
        """

        if not colors:
            colors = cycle(['red', 'darkorange', 'cornflowerblue', 'green', 'yellow'])
        else:
            colors = cycle(colors)

        if self._binary:
            labels = self.labels[::-1]
        else:
            labels = self.labels

        # These next parts are identical to the ROC curve. binarize and then adjust confidences
        binary_ground_truth = label_binarize(self.groundtruth, classes=labels)
        num_classes = binary_ground_truth.shape[1]
        micro_index = num_classes

        data_confidence = []
        for i in range(len(self.groundtruth)):
            # for each of the labels we need to determine the confidence the network had
            data_point_confidence = []
            for c in range(num_classes):
                if self._binary:
                    c = 1
                if self.predicted[i] == labels[c]:
                    data_point_confidence.append(self.confidence[i])
                else:
                    # This acts as a approximation for the confidence. it assumes the probability of the
                    # non-predicted classes to be uniform. I added the max, to prevent div 0 error.
                    data_point_confidence.append((100 - self.confidence[i]) / max(len(labels) - 1, 1))
            data_confidence.append(data_point_confidence)
        data_confidence = np.array(data_confidence)

        # For each class we need to get the precision and recalls in addition to average precision or area
        precision = [None] * (num_classes + 1)
        recall = [None] * (num_classes + 1)
        average_precision = [None] * (num_classes + 1)
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(binary_ground_truth[:, i],
                                                                data_confidence[:, i])
            average_precision[i] = average_precision_score(binary_ground_truth[:, i],
                                                           data_confidence[:, i])

        # Start plotting each of the lines
        fig = plt.figure(figsize=(11, 9))
        lines = []
        plot_labels = []

        # For none binary we will include micro-average, a quantifying score on all classes jointly
        if not self._binary:
            precision[micro_index], recall[micro_index], _ = precision_recall_curve(binary_ground_truth.ravel(),
                                                                                    data_confidence.ravel())
            average_precision[micro_index] = average_precision_score(binary_ground_truth, data_confidence,
                                                                     average="micro")

            l, = plt.plot(recall[micro_index], precision[micro_index], color='navy', linestyle=':', linewidth=2)
            lines.append(l)
            plot_labels.append(f'micro-average Precision-recall (area = {round(average_precision[micro_index], 2)})')

        # add class specific PR curves
        for i, color in zip(range(num_classes), colors):
            ind = i
            if self._binary:
                ind = 1
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append(f'Precision-recall for {labels[ind]} (area = {round(average_precision[i], 2)})')

        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if title:
            plt.title(self.pre_title + title, fontsize=self.title_fontsize)
        else:
            plt.title(self.pre_title + 'Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, plot_labels, loc='lower left', prop=dict(size=14))
        return fig
