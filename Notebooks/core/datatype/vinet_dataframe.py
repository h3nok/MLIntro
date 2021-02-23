from typing import Union

import pandas as pd
import seaborn as sns
from enum import Enum
from collections import Counter
from logs import logger as L
import matplotlib.pyplot as plt
from customer.customer_base import CustomerBase

from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_curve,
                             average_precision_score)
from pandas_profiling import ProfileReport

sns.set()


class viNetDataframeColumn(Enum):
    confidence = 'confidence'
    groundtruth = 'groundtruth'
    predicted = 'predicted'
    inspection_time = 'inspection_time'
    distance = 'distance'
    distance_error = 'distance_error'
    theta = 'theta'
    phi = 'phi'
    tower = 'tower'
    site_id = 'site'
    frame_data = 'frame_data'


@pd.api.extensions.register_dataframe_accessor('vi')
class viNetDataframe:
    _logger = L.Logger('viNetDataframe').configure()

    def __init__(self, df):
        """
            A generic dataframe for viNet reporting
        @param df:
        TOTO - pass pandas dataframe as oppose to a dataprovider
        """
        self._binarized_df = None
        self.__schema__ = 'viNet'
        self._confidence = viNetDataframeColumn.confidence.value
        self._predicted = viNetDataframeColumn.predicted.value
        self._groundtruth = viNetDataframeColumn.groundtruth.value

        self._validate(df)
        self._dataframe = df
        self._customer = None
        self._windfarm = None
        self._tag = 'Unknown Tag'
        self._config = 'Unknown Config'

    def _validate(self, obj):
        """
        viNetDatframe validation - must have groundtruth, predicted and confidence columns
        @param obj:
        @return:
        """
        assert (self._confidence in obj.columns and self._predicted in obj.columns
                and self._groundtruth in obj.columns), \
            "Must at least have '%s', '%s' and '%s' columns." \
            % (self._confidence, self._predicted, self._groundtruth)
        assert len(obj['groundtruth']) == len(obj['predicted']) == len(obj['confidence']), \
            "Dataframe columns are inconsistent "

        if len(obj.index) < 2:
            self._logger.fatal("Stored procedure returned empty dataframe")
            raise RuntimeError("Stored procedure returned empty dataframe")

        self._logger.debug(obj.head)

    @property
    def counts(self):
        return Counter(self._dataframe[self._groundtruth])

    @property
    def binary_counts(self):
        return Counter(self.binarized[self._groundtruth])

    @property
    def customer(self):
        return self._customer

    @property
    def windfarm(self):
        if self.customer is None:
            return None
        if self._windfarm is not None:
            for wf in self.customer.windfarms:
                if wf.name == self._windfarm:
                    return wf
        if len(self.customer.windfarms) == 0:
            return None
        else:
            return self.customer.windfarms[0]

    @property
    def config(self):
        return self._config

    @property
    def tag(self):
        return self._tag

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def columns(self):
        return list(self._dataframe.columns)

    @property
    def head(self):
        return self._dataframe.head

    @property
    def classifications(self):
        return sorted(list(set(self._dataframe[self._groundtruth])))

    @property
    def protected_classifications(self):
        """
        Returns list of protected classes or None if no customer is supplied. Customer is needed to get protected
        """
        if self.customer is None:
            return None
        return sorted([b.name for b in self._customer.protected_species if b.deployed])

    @property
    def binary_classifications(self):
        """ reverse is set to true so 'protected' is the first element"""
        return sorted(list(set(self.binarized[self._groundtruth])), reverse=True)

    @property
    def binarized(self):
        """
        Once binarized, this can be used to access the new df
        """
        if self._binarized_df is None:
            protected_classes = self.protected_classifications
            if protected_classes is not None:
                self.binarize(protected_classes)
            else:
                self.binarize([])
        return self._binarized_df

    def binarize(self, category_0, label_0='Protected', label_1='Other') -> None:
        """
            Convert groundtruth and predicted labels to binary
        @param category_0: original labels that correspond to label_0, TOTW ex.  [Bald-Eagle, Golden-Eagle]
        @param label_0: a label for grouping all categories in category_0, ex. 'Protected'
        @param label_1: species that are not part of category_0 will be labeled with this, ex. 'Other'l
        @return:
        """
        if category_0 is None:
            protected_classes = self.protected_classifications
            if protected_classes is not None:
                self.binarize(protected_classes)
            else:
                return
        if self._binarized_df is None:
            self._binarized_df = dict(self._dataframe)
            self._binarized_df[self._groundtruth] = [label_0 if (item in category_0) else label_1 for
                                                     item in self._dataframe[self._groundtruth]]

            self._binarized_df[self._predicted] = [label_0 if (item in category_0) else label_1 for
                                                   item in self._dataframe[self._predicted]]

    def set_attributes(self, customer, **kwargs):
        """ On creation using viNetResultDataProvider.dataframe the tag and config are automatically set """
        assert customer is None or isinstance(customer, CustomerBase)
        self._customer = customer
        if 'windfarm_name' in kwargs.keys():
            self._windfarm = kwargs['windfarm_name']
        if 'tag' in kwargs.keys():
            self._tag = kwargs['tag']
        if 'config' in kwargs.keys():
            self._config = kwargs['config']

    def plot(self, plot_type):
        from core.datatype.vinet_dataframe_plotter import DataFramePlotter
        print("Use DataFramePlotter.plot instead. plotter class found in core.datatype.vinet_dataframe_plotter")
        # Although this class if no longer supported it will still call the plotter class
        plotter = DataFramePlotter(self)
        # Not that only basic plots are supported. so only things like cm, pr, roc ... Not b-cm or subdivisions
        return plotter.plot(plot_type)

    def calc_metrics(self, metric_types: Union[str, list] = 'all'):
        """
        Calculates different metrics from the dataframe

        :param metric_types: a list of the metrics to be calculate.
            Options: `['accuracy', 'fnr', 'fpr', 'tnr', 'tpr', 'binary_accuracy']`
            Or use 'all' (not in a list) to calculate all metrics.
        :return: dict of metrics requested;
        """
        options = ['accuracy', 'fnr', 'fpr', 'tnr', 'tpr', 'binary_accuracy']

        if isinstance(metric_types, str):
            if metric_types.lower() == 'all':
                metric_types = options
            else:
                metric_types = [metric_types]

        for t in metric_types:
            assert isinstance(t, str)
            assert t.lower() in options, '`metric_types` item {} must be in {}'.format(t, options)

        metric_types = list(map(lambda t: t.lower(), metric_types))

        ground_truth = []
        predicted = []
        labels = []
        bgt = []
        bpred = []
        bl = None

        ret = dict()

        if options[0] in metric_types:  # accuracy
            if len(ground_truth) == 0:
                ground_truth = list(self._dataframe[self._groundtruth])
            if len(predicted) == 0:
                predicted = list(self._dataframe[self._predicted])

            ret[options[0]] = accuracy_score(ground_truth, predicted)  # accuracy

        if options[1] in metric_types or options[2] in metric_types \
                or options[3] in metric_types or options[4] in metric_types:  # fnr, fpr, tnr, tpr
            assert self._binarized_df is not None, 'You need to call binarize() before you call calc_metrics'
            if len(bgt) == 0 or len(bpred) or bl in None:
                bgt = list(self.binarized[self._groundtruth])
                bpred = list(self.binarized[self._predicted])
                bl = self.binary_classifications
                bl.reverse()

            tn, fp, fn, tp = confusion_matrix(bgt, bpred, labels=bl).ravel()
            ret[options[4]] = (tp / (tp + fn))  # tpr
            ret[options[3]] = (tn / (tn + fp))  # tnr
            ret[options[2]] = (fp / (fp + tn))  # fpr
            ret[options[1]] = (fn / (fn + tp))  # fnr

        if options[5] in metric_types:  # binary accuracy
            if len(bgt) == 0 or len(bpred) or bl in None:
                bgt = list(self.binarized[self._groundtruth])
                bpred = list(self.binarized[self._predicted])

            ret[options[5]] = accuracy_score(bgt, bpred)  # binary accuracy

        return ret

    def profile_report(self, save_loc=None, title="Pandas Profiling Report"):
        self._logger.info("Generating Pandas Profile Report ... ")
        profile = ProfileReport(self._binarized_df, title=title, explorative=True)
        profile.to_file(save_loc)
        self._logger.info("Profile report saved to {}".format(save_loc))


