import pandas as pd
import seaborn as sns

from core.datatype.vinet_dataframe import viNetDataframe
from core.datatype.vinet_dataframe import viNetDataframeColumn
from core.image.frame_attribute import FrameAttribute
from core.plots.confusion_matrix import ConfusionMatrix
from core.plots.distribution import Distribution
from core.plots.line_plot import LinePlot
from core.plots.plot_data import viNetValidationData
from core.plots.roc_curve import ROC

sns.set_palette("husl")


class DataFramePlotter:
    """
    viNet results dataframe reports
    """
    __vinet_df = None

    _plot_data = None
    _pd_subdivisions = None

    _binary_allowed = False
    _binary_plot_data = None
    _bpd_subdivisions = None

    _images = None

    _inspection_time_label = None

    def __init__(self, df, colors=None):
        """
        Takes in the dataframe and can then plot anything the user wants. To use binary plots. Provide customer to df
        @param df: dataframe object with classification data, can be either viNetDataframe or pd.core.frame.DataFrame
        Must have column, Ground Truth, Predicted, Confidence. And inspection time if time based plot is used. and
        Frame Data for frame attribute based plots
        """

        if isinstance(df, viNetDataframe):
            self.__vinet_df = df
        elif isinstance(df, pd.core.frame.DataFrame):
            self.__vinet_df = df.vi
        else:
            raise TypeError(f"Type of df not allowed: {type(df)}")

        self.__colors = colors
        # From dataframe we need to create our plot_data objects for both binary and non binary
        self._inspection_time_label = viNetDataframeColumn.inspection_time.value
        gt_label = viNetDataframeColumn.groundtruth.value
        pred_label = viNetDataframeColumn.predicted.value

        conf_lst = list(self.__vinet_df.dataframe[viNetDataframeColumn.confidence.value])
        inspec_lst = list(self.__vinet_df.dataframe[self._inspection_time_label])

        self._plot_data = viNetValidationData(ground_truth=list(self.__vinet_df.dataframe[gt_label]),
                                              predicted=list(self.__vinet_df.dataframe[pred_label]),
                                              confidence=conf_lst, labels=self.__vinet_df.classifications,
                                              inspection_time=inspec_lst)
        if self.__vinet_df.protected_classifications is not None:
            self._binary_plot_data = viNetValidationData(ground_truth=list(self.__vinet_df.binarized[gt_label]),
                                                         predicted=list(self.__vinet_df.binarized[pred_label]),
                                                         confidence=conf_lst, labels=self.__vinet_df.binary_classifications,
                                                         inspection_time=inspec_lst)
            self._binary_allowed = True

        # If images were in the dataframe we will also grab them
        if viNetDataframeColumn.frame_data.value in self.__vinet_df.dataframe.columns:
            self._images = self.__vinet_df.dataframe[viNetDataframeColumn.frame_data.value]

        # Dictionaries to hold subdivisions. This might get costly for how much data we save.
        self._pd_subdivisions = dict()
        if self._binary_allowed:
            self._bpd_subdivisions = dict()

    def get_subdivision_data(self, binary, subdivision=None, subdiv_option=None):
        """
        Takes the original plot_data and returns a subdivision of the data, for example all classifications in the
        winter. Or all classification with low contrast. All subdivisions are in the for of PlotData and will contain
        the plot_data and a title prefix.
        @param binary: get binary classification data
        @param subdivision: str for subdivision, ex. month, season, weather, brightness
        @param subdiv_option: str for option (optional but sometimes required) ex. january, winter, rainy, low
        """
        # if no subdivisions are requested then we will ignore the subdivisions entirely and return the plot_data
        if subdivision is None:
            if binary and self._binary_allowed:
                return self._binary_plot_data
            else:
                return self._plot_data
        # to keep our dictionaries clean we will have a default option
        if subdiv_option is None:
            subdiv_option = "default"

        subdivision = subdivision.lower()
        subdiv_option = subdiv_option.lower()

        # If subdivision is requested then we need to see get that data either from already saved dicts or create it
        if binary and self._binary_allowed:
            data = self._binary_plot_data
            subdivision_data = self._bpd_subdivisions
        else:
            data = self._plot_data
            subdivision_data = self._pd_subdivisions

        # if the data does not exists we will create it.
        if subdivision not in subdivision_data or subdivision_data[subdivision] is None or \
                subdiv_option not in subdivision_data[subdivision]:
            # create data and then add data
            if subdivision not in subdivision_data:
                subdivision_data[subdivision] = dict()
            if subdivision in ["month", "season", "sun"]:  # Time based subdivisions
                subdivision_data[subdivision] = data.create_time_subdivision(self._inspection_time_label, subdivision,
                                                                             self.__vinet_df.windfarm)

            elif subdivision in ["rain"]:  # weather based subdivisions
                subdivision_data[subdivision] = data.create_weather_subdivision(self._inspection_time_label,
                                                                                subdivision, self.__vinet_df.windfarm)

            elif subdivision in dir(FrameAttribute):  # FrameAttribute subdivisions
                top = True  # if True the subdivision will grab values with the highest values for the FA
                if subdiv_option is not None and (subdiv_option == 'bottom' or subdiv_option == 'low'):
                    top = False  # If False then it will grab the lowest values
                subdivision_data[subdivision][subdiv_option] = data.create_frame_attribute_subdivision(
                    self._images, subdivision, subdiv_option, top=top, inplace=False)

            else:
                print(f"Not a valid subdivision: {subdivision}.")
                return None
        # Once data has been saved we will return the plot_data
        return subdivision_data[subdivision].get_persisted_data(subdiv_option, None)

    def plot(self, plot_type, binary=False, subdivision=None, subdiv_option=None):
        """
        Plot desired graph with desired subdivisions
        @param plot_type: Type of plot as str. ex 'cm', 'roc', 'pr', 'time', 'cor-brightness'
        @param binary: create binary classification plot
        @param subdivision: Optional, desired subdivision as string
        @param subdiv_option: Optional option for subdivision as string
        """
        # First get subdivision data
        # This will also create a pre_title if needed
        if isinstance(subdivision, list):
            raise NotImplementedError("Multiple subdivisions not supported")

        data = self.get_subdivision_data(binary, subdivision, subdiv_option)

        if data is None or len(data.ground_truth) < 10:
            print("get_subdivision_data returned None skipping plot")
            return None

        # Distributions. Scatter and hist do not have frame attribute subdivision support, or binary support
        if 'scatter' in plot_type:  # Here you can do both scatter or correct_scatter
            scatter = Distribution.ScatterPlot(data, title="")
            if 'correct' in plot_type:
                # Correct will only display correctly predicted classification and their confidence.
                return scatter.plot_categorical_scatter_plot("Categorical Scatter Plot For Correct Predictions", True)
            else:
                return scatter.plot_categorical_scatter_plot("Categorical Scatter Plot")
        elif 'hist' in plot_type:
            return sns.countplot(x=data.ground_truth, data=self.__vinet_df)
        elif 'pie' in plot_type:
            title = "Frame Distribution"
            return Distribution.Pie(list(data.counts.values()),
                                    list(data.counts.keys()),
                                    title=data.pre_title + title).plot()
        elif 'kde' in plot_type:
            kde = Distribution.KDE(data)
            if 'multi' in plot_type:
                title = "Class based KDE of network confidence"
                return kde.plot_multiclass(title=title)
            else:
                title = "KDE of network confidence"
                return kde.plot(title=title)

        # Performance Metrics
        if 'cm' in plot_type:
            cm = ConfusionMatrix(data)
            return cm.plot_confusion_matrix(title="Confusion Matrix of Network Classification", color=self.__colors)
        elif 'cr' in plot_type:
            cm = ConfusionMatrix(data)
            return cm.plot_classification_report(title="Classification Report for Network", color=self.__colors)
        elif 'roc' in plot_type:
            roc = ROC(data)
            return roc.plot(title="ROC Curve for Network")
        elif 'time' in plot_type:
            line_plot = LinePlot(data)
            return line_plot.plot_time_correlation("Network Confidence vs Time of Day",
                                                   list(data.other['inspection_time']))
        elif 'cor-' in plot_type:
            # the supplied string plot_type needs to be cor-<function from frame_attribute>-<optional option> for
            # example cor-entropy or cor-brightness, or cor-average_color-blue
            if self._images is None:
                print("Image data not provided within data frame. Can not create frame attribute correlation plot.")
                return None

            # Parse function name and option
            split_plot_type = plot_type.split('-')
            fa_func = split_plot_type[1]
            if len(split_plot_type) > 2:
                fa_option = split_plot_type[2]
            else:
                fa_option = None

            line_plot = LinePlot(data)
            return line_plot.plot_frame_attribute_correlation(f"Network Confidence vs {fa_func.replace('_', ' ')}",
                                                              self._images, fa_func, fa_option=fa_option)
        elif 'pr' in plot_type:
            cm = ConfusionMatrix(data)
            return cm.plot_precision_recall(title="Precision-Recall Curve")
        else:
            print(f"Plot not found: {plot_type}. Skipping plot")
            return None

