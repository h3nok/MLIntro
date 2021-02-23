from reporting.report_generator import Report


class ClassifierPerformanceReport(Report):
    """
        Neural network performance plots
    """

    def __init__(self, data, pdf_gen=None, quality="normal", colors=None):
        """
        This class focuses on the entire network. Tries to plot the date to show overall trends and non-specialized
        cases.
        @param data: dataframe object to create plots from
        @param pdf_gen: an outside pdf generator can be provided.
        @param quality: can be string "low", "normal", "high" or int for the dpi value
        """
        super().__init__(data, title="viNet Performance report", pdf_gen=pdf_gen, quality=quality, colors=colors)

    def confusion_matrix(self, binary=False):
        self.plot_by_str('cm', binary)

    def roc(self, binary=False):
        self.plot_by_str('roc', binary)

    def classification_report(self, binary=False):
        self.plot_by_str('cr', binary)

    def pr_curve(self, binary=False):
        self.plot_by_str('pr', binary)

    @staticmethod
    def create_complete_report(data, save_loc, appendix=False, colors=None):
        """
        Automatic method of creating report from dataframe. Creates correlation plots
        @param colors:
        @param data: dataframe, needs to have a customer saved to properly create binary plots
        @param save_loc: location to save pdf
        @param appendix: create appendix page
        """
        # this will create the title page
        report_gen = ClassifierPerformanceReport(data, colors=colors)

        # create Pie charts
        report_gen.plot_by_str('pie')
        report_gen.plot_by_str('pie', binary=True)

        report_gen.plot_by_str('complete_scatter')
        # report_gen.plot_by_str('scatter')
        report_gen.plot_by_str('multikde')

        report_gen.confusion_matrix(binary=True)
        report_gen.classification_report(binary=True)
        report_gen.confusion_matrix(binary=False)
        report_gen.classification_report(binary=False)

        report_gen.roc(binary=False)

        report_gen.pr_curve(binary=False)

        # This will create approval and appendix pages
        report_gen.save_as_pdf(save_loc, appendix=appendix, open_pdf=True)
        # data.vi.profile_report(save_loc + ".html")
