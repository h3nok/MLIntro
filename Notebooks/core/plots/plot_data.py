from core.image.frame_attribute import FrameAttribute
from PIL import Image
import io
import numpy as np
from collections import Counter
from tools.weather import Weather
from datetime import timezone, timedelta


MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
SEASONS = ["Summer", "Fall", "Winter", "Spring"]
SUN_TIMES = ["Sunset", "Sunrise", "Midday"]
RAIN_ARR = ["Rainy", "Possible Rain", "No Rain"]


class viNetValidationData:
    _ground_truth = None
    _predicted = None
    _confidence = None
    _labels = None
    _other = None

    _title_prefix = ""

    def __init__(self, ground_truth, predicted, confidence, labels, **kwargs):
        """
        Not all plots require these 4 items so any or all of them can be None
        @param ground_truth: ground truth array
        @param predicted: predicted array
        @param confidence: confidence array
        @param labels: labels
        @param kwargs: addition arrays
        """
        self._ground_truth = ground_truth
        self._predicted = predicted
        self._labels = labels
        self._confidence = confidence

        self._other = kwargs

        self._verify()

    def _verify(self):
        """ Makes sure things are of the same len """
        if self._ground_truth and self._predicted:
            assert len(self._ground_truth) == len(self._predicted)
        if self._confidence is not None and self._ground_truth:
            assert len(self._confidence) == len(self._ground_truth)
        if self._ground_truth is not None and self._other is not None:
            for k in self._other.keys():
                assert len(self.ground_truth) == len(self._other[k])

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def predicted(self):
        return self._predicted

    @property
    def confidence(self):
        return self._confidence

    @property
    def labels(self):
        return self._labels

    @property
    def counts(self):
        return Counter(self._ground_truth)

    @property
    def pre_title(self):
        return self._title_prefix

    def set_title_prefix(self, val):
        self._title_prefix = val

    @property
    def other(self):
        return self._other

    def create_frame_attribute_subdivision(self, images, fa_func, fa_option=None, thresh=None, top=True, inplace=True):
        """
        Given a frame attribute function and images this will create a subdivision of the elements that are above
        a threshold or in the 75% or above percentile
        @param images: images array as bytes (default from db)
        @param fa_func: function name from frame_attribute
        @param fa_option: an option to help make fa_func a numeric value
        @param thresh: threshold of which to get images above
        @param top: pick images above (true) or below (false)
        @param inplace: either change this objects data or return a new object. suggested =False to return data
        """
        assert images is not None

        # images must be the same size of what ever array we have
        if self._ground_truth is not None:
            assert len(images) == len(self._ground_truth)
        if self._predicted is not None:
            assert len(images) == len(self._predicted)
        if self._confidence is not None:
            assert len(images) == len(self._confidence)
        if self._other is not None:
            for k in self._other.keys():
                assert len(images) == len(self._other[k])

        fa_scores = []
        for i in range(len(images)):
            img = np.array(Image.open(io.BytesIO(images[i])))[..., :3]
            fa = FrameAttribute(img)
            fa_result = fa.attribute_to_single_val(fa_func, fa_option)
            fa_scores.append(fa_result)

        if thresh is None:
            if top:
                thresh = np.percentile(fa_scores, 75)
            else:
                thresh = np.percentile(fa_scores, 25)
        if top:
            title_prefix = f"High {fa_func.replace('_', ' ')}: "
        else:
            title_prefix = f"Low {fa_func.replace('_', ' ')}: "

        if fa_func == FrameAttribute.dominant_colors.__name__ or fa_func == FrameAttribute.average_color.__name__:
            title_prefix = title_prefix[:-2] + f"({fa_option}ness): "

        new_gt = []
        new_pred = []
        new_conf = []
        new_other = {}
        if self._other is not None:
            for k in self._other.keys():
                new_other[k] = []
        for i in range(len(fa_scores)):
            if (fa_scores[i] <= thresh and not top) or (fa_scores[i] >= thresh and top):
                if self._ground_truth is not None:
                    new_gt.append(self._ground_truth[i])
                if self._predicted is not None:
                    new_pred.append(self._predicted[i])
                if self._confidence is not None:
                    new_conf.append(self._confidence[i])
                if self._other is not None:
                    for k in self._other.keys():
                        new_other[k].append(self._other[k][i])

        if self._ground_truth is not None and self._labels is not None:
            new_labels = sorted(list(set(new_gt)), key=lambda bird: self.labels.index(bird))
        else:
            new_labels = self._labels

        if inplace:
            self._labels = new_labels
            self._title_prefix = title_prefix

            if self._ground_truth is not None:
                self._ground_truth = new_gt
            if self._predicted is not None:
                self._predicted = new_pred
            if self._confidence is not None:
                self._confidence = new_conf
            if self._other is not None:
                self._other = new_other

            self._verify()
        else:
            new_data = viNetValidationData(new_gt, new_pred, new_conf, new_labels, **new_other)
            new_data.set_title_prefix(title_prefix)
            return new_data

    # TODO: create third function to account for the overlap of these two functions. I am going to wait to do this.
    def create_time_subdivision(self, inspection_time_label, option="month", windfarm=None):
        """
        Given an option, ex month. this will create a dictionary of the subdivisions for each month
        @param inspection_time_label: label in self.other for inspection times
        @param option: string, month or season, possible yearly too
        @param windfarm:
        """
        assert inspection_time_label in self._other.keys()

        if option == 'month':
            key_arr = MONTHS
        elif option == 'season':
            if windfarm is None:
                return None
            key_arr = SEASONS
            weather = Weather(windfarm.latitude, windfarm.longitude)
        elif option == 'sun':
            if windfarm is None:
                return None
            key_arr = SUN_TIMES
            weather = Weather(windfarm.latitude, windfarm.longitude)
        else:
            return None

        new_gt = []
        new_pred = []
        new_conf = []
        new_other = {}
        if self._other is not None:
            for k in self._other.keys():
                new_other[k] = []

        for _ in key_arr:
            # Add each month in the keys under MONTH
            new_gt.append([])
            new_pred.append([])
            new_conf.append([])
            if self._other is not None:
                for k in self._other.keys():
                    new_other[k].append([])

        for i in range(len(self._other[inspection_time_label])):
            # Get key for key_arr
            key = 0
            if option == 'month':
                key = int(self._other[inspection_time_label][i].month) - 1
            elif option == 'season':
                key = SEASONS.index(weather.get_season(self._other[inspection_time_label][i]))
            elif option == 'sun':
                dt = self._other[inspection_time_label][i].replace(tzinfo=windfarm.timezone)
                utc_dt = dt.astimezone(timezone.utc)
                sr, ss = weather.get_sunrise_sunset(date=dt, time_zone=0)

                if (utc_dt - sr) < timedelta(seconds=3600):
                    key = 1
                elif (ss - utc_dt) < timedelta(seconds=3600):
                    key = 0
                else:
                    key = 2

            if self._ground_truth is not None:
                new_gt[key].append(self._ground_truth[i])
            if self._predicted is not None:
                new_pred[key].append(self._predicted[i])
            if self._confidence is not None:
                new_conf[key].append(self._confidence[i])
            if self._other is not None:
                for k in self._other.keys():
                    new_other[k][key].append(self._other[k][i])

        new_data = dict()
        for key_ind in range(len(key_arr)):
            other_dict = dict()
            if new_other is not None:
                for k in new_other.keys():
                    other_dict[k] = new_other[k][key_ind]
            if self._ground_truth is not None and self._labels is not None:
                new_labels = sorted(list(set(new_gt[key_ind])), key=lambda bird: self.labels.index(bird))
            else:
                new_labels = self._labels
            new_data[key_arr[key_ind].lower()] = viNetValidationData(new_gt[key_ind], new_pred[key_ind], new_conf[key_ind],
                                                                     new_labels, **other_dict)
            new_data[key_arr[key_ind].lower()].set_title_prefix(f"{key_arr[key_ind]}: ")

        return new_data

    def create_weather_subdivision(self, inspection_time_label, option, windfarm):
        """
            Given an option, ex month. this will create a dictionary of the subdivisions for each month
            @param inspection_time_label: label in self.other for inspection times
            @param option: string, month or season, possible yearly too
            @param windfarm:
            """
        assert inspection_time_label in self._other.keys()

        weather = Weather(windfarm.latitude, windfarm.longitude)

        if option == "rain":
            key_arr = RAIN_ARR
        else:
            return None

        new_gt = []
        new_pred = []
        new_conf = []
        new_other = {}
        if self._other is not None:
            for k in self._other.keys():
                new_other[k] = []

        for _ in key_arr:
            # Add each month in the keys under MONTH
            new_gt.append([])
            new_pred.append([])
            new_conf.append([])
            if self._other is not None:
                for k in self._other.keys():
                    new_other[k].append([])

        for i in range(len(self._other[inspection_time_label])):
            # Get key for key_arr
            key = 0
            if option == 'rain':
                rain = weather.get_rain(self._other[inspection_time_label][i])
                key = 2
                if rain > .6:
                    key = 0
                elif rain > .3:
                    key = 1

            if self._ground_truth is not None:
                new_gt[key].append(self._ground_truth[i])
            if self._predicted is not None:
                new_pred[key].append(self._predicted[i])
            if self._confidence is not None:
                new_conf[key].append(self._confidence[i])
            if self._other is not None:
                for k in self._other.keys():
                    new_other[k][key].append(self._other[k][i])

        new_data = dict()
        for key_ind in range(len(key_arr)):
            other_dict = dict()
            if new_other is not None:
                for k in new_other.keys():
                    other_dict[k] = new_other[k][key_ind]
            if self._ground_truth is not None and self._labels is not None:
                new_labels = sorted(list(set(new_gt[key_ind])), key=lambda bird: self.labels.index(bird))
            else:
                new_labels = self._labels
            new_data[key_arr[key_ind].lower()] = viNetValidationData(new_gt[key_ind], new_pred[key_ind], new_conf[key_ind],
                                                                     new_labels, **other_dict)
            new_data[key_arr[key_ind].lower()].set_title_prefix(f"{key_arr[key_ind]}: ")

        return new_data
