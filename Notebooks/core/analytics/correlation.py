from core.plots.plot_data import MONTHS, SEASONS, SUN_TIMES, RAIN_ARR
from core.analytics.reporting.report_generator import Report


class ClassifierCorrelationPlots(Report):
    """
        Neural network correlation plots
    """

    def __init__(self, data, pdf_gen=None, quality="normal", colors=None):
        """
        This class implements Reporter to utilize a few of its functions. But this classes focuses
        on correlations, and specialized cases. It compares frame attributes to confidence and time of year and day to
        accuracy.
        @param data: dataframe object to create plots from. Dataframe should have a customer object set
        @param pdf_gen: an outside pdf generator can be provided.
        @param quality: can be string "low", "normal", "high" or int for the dpi value
        :return: two string representing a subdivision and a subdivision option for plot_data.
        """
        super().__init__(data, title="viNet Correlation report", pdf_gen=pdf_gen, quality=quality, colors=colors)

    @staticmethod
    def _parse_timeweather_str(tw_str):
        if tw_str is not None:
            tw_str = tw_str.lower()

        if tw_str is None:
            yield None, None
        elif tw_str in "months":
            for m in MONTHS:
                yield 'month', m
        elif tw_str in [m.lower() for m in MONTHS]:
            yield 'month', tw_str
        elif tw_str in "seasons":
            for s in SEASONS:
                yield "season", s
        elif tw_str in [s.lower() for s in SEASONS]:
            yield 'season', tw_str
        elif tw_str in "sun_times":
            for s in SUN_TIMES:
                yield "sun", s
        elif tw_str in [s.lower() for s in SUN_TIMES]:
            yield "sun", tw_str
        elif tw_str in "rain":
            for r in RAIN_ARR:
                yield "rain", r
        elif tw_str in [r.lower() for r in RAIN_ARR]:
            yield "rain", tw_str
        else:
            yield None, None

    def confusion_matrix(self, binary=False, time_or_weather=None):
        """ Plot confusion matrix. time should be a string with desired subdivision such as "January" for the month
        of january. Or "Winter" for winter or even "Months" for all possible months. This applies to all following
        functions"""
        for subdiv, subdiv_op in self._parse_timeweather_str(time_or_weather):
            self.plot_by_str('cm', binary, subdiv, subdiv_op)

    def roc(self, binary=False, time_or_weather=None):
        for subdiv, subdiv_op in self._parse_timeweather_str(time_or_weather):
            self.plot_by_str('roc', binary, subdiv, subdiv_op)

    def classification_report(self, binary=False, time_or_weather=None):
        for subdiv, subdiv_op in self._parse_timeweather_str(time_or_weather):
            self.plot_by_str('cr', binary, subdiv, subdiv_op)

    def pr_curve(self, binary=False, time_or_weather=None):
        for subdiv, subdiv_op in self._parse_timeweather_str(time_or_weather):
            self.plot_by_str('pr', binary, subdiv, subdiv_op)

    def confidence_correlation(self, attribute, binary=False):
        """
        @param attribute: must be a frame_attribute function name, and optional parameter, or "time". Ex. "brightness",
        "time", "average_color-red"
        @param binary:
        """
        if attribute == 'time':
            self.plot_by_str('time', binary)
        else:
            self.plot_by_str(f'cor-{attribute}', binary)

    @staticmethod
    def create_complete_report(data, save_loc=None, appendix=False, colors=None):
        """
        Automatic method of creating report from dataframe. either provide data or tag config and classification group
        @param colors:
        @param data: dataframe, needs to have a customer saved to properly create binary plots. Must also have images
        @param save_loc: location to save pdf
        @param appendix: create appendix page
        """
        report_gen = ClassifierCorrelationPlots(data, colors=colors)

        report_gen.plot_by_str('pie')

        # not all the functions play nicely so it easier to do them one at a time.
        from core.image.frame_attribute import FrameAttribute
        report_gen.confidence_correlation(attribute='time')
        report_gen.confidence_correlation(attribute=FrameAttribute.brightness.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.perceived_brightness.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.noise.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.sharpness.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.variance.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.fourier_spectrum_iq.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.contrast.__name__)
        report_gen.confidence_correlation(attribute=FrameAttribute.entropy.__name__)
        report_gen.confidence_correlation(attribute=f'{FrameAttribute.average_color.__name__}-blue')
        report_gen.confidence_correlation(attribute=f'{FrameAttribute.average_color.__name__}-red')
        report_gen.confidence_correlation(attribute=f'{FrameAttribute.average_color.__name__}-green')

        # Monthly break down using confusion matrices
        report_gen.confusion_matrix(binary=False, time_or_weather='months')

        # Seasonal break down using confusion matrices
        report_gen.confusion_matrix(binary=False, time_or_weather='seasons')

        # Sunlight break down. Sunrise / Sunset / Midday
        report_gen.confusion_matrix(binary=False, time_or_weather='sun')

        # This will create approval and appendix pages
        report_gen.save_as_pdf(save_loc, appendix=appendix, open_pdf=True)

    @staticmethod
    def create_fine_grained_report(data, save_loc, appendix):
        """
        THis function will create plots to highlight fine grained report on time and weather. Note that weather plots
        can be incorrect. It not entirely possible to determine the weather at a location.
        """

        report_gen = ClassifierCorrelationPlots(data)

        # Monthly break down using confusion matrices
        report_gen.confusion_matrix(binary=False, time_or_weather='months')
        report_gen.roc(binary=False, time_or_weather='months')

        # Seasonal break down using confusion matrices
        report_gen.confusion_matrix(binary=False, time_or_weather='seasons')
        report_gen.roc(binary=False, time_or_weather='seasons')

        # Sunlight break down. Sunrise / Sunset / Midday
        report_gen.confusion_matrix(binary=False, time_or_weather='sun')
        report_gen.roc(binary=False, time_or_weather='sun')

        # weather
        report_gen.confusion_matrix(binary=False, time_or_weather="rain")
        report_gen.roc(binary=False, time_or_weather="rain")

        # This will create approval and appendix pages
        report_gen.save_as_pdf(save_loc, appendix=appendix)
