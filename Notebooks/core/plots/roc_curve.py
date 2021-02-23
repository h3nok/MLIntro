import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, roc_curve, log_loss)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from core.plots.plot_base import viNetPlotBase


class ROC(viNetPlotBase):
    _macro_index = None
    _binary = False

    def __init__(self, plot_data):
        """
        @param plot_data: Must include
        ground_truth: array of ground truth data
        predicted: corresponding predictions
        confidence: corresponding confidence of prediction
        labels: labels of the ground truth and prediction lists. Order matters if there are only two labels. See plot()
        """
        super().__init__(plot_data)

        self._labels = self.labels
        if len(self._labels) == 2:
            self._binary = True
            self._labels = self.labels[::-1]

    def plot(self, title="", colors=None):
        """
        Generates roc curves for each provided label.
        If only two labels are provided it will create only one curve, for the first label
        @param title: Title of plot
        @param colors: colors array for each of the roc curves
        @return: figure of plot
        @todo: default bi colors
        """
        plt.style.use('seaborn-darkgrid')

        if not colors:
            colors = cycle(['red', 'darkorange', 'cornflowerblue', 'green', 'yellow'])
        else:
            colors = cycle(colors)

        # Binarize the output
        binary_ground_truth = label_binarize(self.groundtruth, self._labels)
        num_classes = binary_ground_truth.shape[1]
        self._macro_index = num_classes

        data_confidence = []
        for i in range(len(self.groundtruth)):
            # for each of the labels we need to determine the confidence the network had
            data_point_confidence = []
            for c in range(num_classes):
                if self._binary:
                    c = 1
                if self.predicted[i] == self._labels[c]:
                    data_point_confidence.append(self.confidence[i])
                else:
                    # This acts as a approximation for the confidence. it assumes the probability of the
                    # non-predicted classes to be uniform. I added the max, to prevent div 0 error.
                    data_point_confidence.append((100 - self.confidence[i])/max(len(self._labels)-1, 1))
            data_confidence.append(data_point_confidence)
        data_confidence = np.array(data_confidence)

        # Compute macro-average ROC curve and ROC area
        # Compute ROC curve and ROC area for each class
        # We will create a different ROC curve for each label
        fpr = [None]*(num_classes + 1)
        tpr = [None]*(num_classes + 1)
        roc_auc = [None]*(num_classes + 1)
        l_loss = [None]*(num_classes + 1)
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(binary_ground_truth[:, i], data_confidence[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            l_loss[i] = log_loss(binary_ground_truth, data_confidence)

        # Create figure
        fig = plt.figure(figsize=(11, 9))

        if not self._binary:
            # This will generate the macro average. but we only want this if its not a binary case
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

            # Then interpolate all ROC curves
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= num_classes

            fpr[self._macro_index] = all_fpr
            tpr[self._macro_index] = mean_tpr
            roc_auc[self._macro_index] = auc(all_fpr, mean_tpr)

            # Plot Macro average ROC curve
            plt.plot(fpr[self._macro_index], tpr[self._macro_index],
                     label=' ROC curve (area = {0:0.3f})'.format(roc_auc[self._macro_index]),
                     color='navy', linestyle=':', linewidth=4)

        for i, color in zip(range(num_classes), colors):
            ind = i
            if self._binary:
                ind = 1
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve for {0} (area = {1:0.3f})'.format(self._labels[ind], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.pre_title + title, fontsize=self.title_fontsize)
        plt.legend(loc="lower right")

        return fig
