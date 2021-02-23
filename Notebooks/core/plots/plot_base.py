from collections import Counter
import pandas as pd
from core.plots.plot_data import viNetValidationData


class viNetPlotBase:
    __plot_data = None

    def __init__(self, data):
        print(type(data))
        if isinstance(data, (pd.DataFrame, viNetValidationData, dict)):
            self.__plot_data = data
        else:
            raise TypeError("param plot_data must be of type PlotData")

    @property
    def data(self):
        return self.__plot_data

    @property
    def groundtruth(self):
        return self.__plot_data.ground_truth

    @property
    def predicted(self):
        return self.__plot_data.predicted

    @property
    def confidence(self):
        return self.__plot_data.confidence

    @property
    def labels(self):
        return self.__plot_data.labels

    @property
    def counts(self):
        return Counter(self.groundtruth)

    @property
    def pre_title(self):
        return self.__plot_data.pre_title

    @property
    def title_fontsize(self):
        return 20
