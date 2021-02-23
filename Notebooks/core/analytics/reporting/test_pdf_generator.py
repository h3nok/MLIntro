from core.analytics.reporting.pdf_generator import PDFGenerator
from core.datatype.vinet_dataframe_plotter import DataFramePlotter
from unittest import TestCase


class TestPDFGenerator(TestCase):
    def test_generate_from_plots(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        from providers.reportdata_provider import viNetResultDataProvider
        with viNetResultDataProvider(tag, config,
                                     classification_group=cgroup,
                                     procedure=sproc, include_images=True) as dp:
            dp.set_limit(10000)
            dp.fetch_results()
            vinet_datum = dp.dataframe

            plotter = DataFramePlotter(vinet_datum)

            figs = [plotter.plot('cm'),
                    plotter.plot('pr'),
                    plotter.plot('cr'),
                    plotter.plot('roc'),
                    plotter.plot('time'),
                    plotter.plot('pie')]

            PDFGenerator().generate_from_plots('test_report.pdf', config, tag, figs)

