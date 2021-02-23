from core.analytics.performance import ClassifierPerformanceReport
from customer.customers import CustomerObjectMap


class TestClassifierPerformancePlots:
    def test_plots(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        # although this is greyed out this is still a required import
        from providers.reportdata_provider import viNetResultDataProvider

        with viNetResultDataProvider(tag, config,
                                     classification_group=cgroup,
                                     procedure=sproc) as dp:
            # dp.set_limit(10000)
            dp.fetch_results()
            vinet_datum = dp.dataframe

            # Customer and windfarm set.
            # Customer is needed for protected classes. windfarm name needs to be the name of one of the windfarms.
            # and will tell the functions the location of the windfarm.
            vinet_datum.vi.set_attributes(customer=CustomerObjectMap["Vattenfall"]())

            # this will create the title page
            report_gen = ClassifierPerformanceReport(vinet_datum)

            # create Pie charts
            report_gen.plot_by_str('pie')
            report_gen.plot_by_str('pie', binary=True)

            report_gen.plot_by_str('scatter')
            report_gen.plot_by_str('correct_scatter')
            report_gen.plot_by_str('kde')
            report_gen.plot_by_str('multikde')

            report_gen.confusion_matrix(binary=True)
            report_gen.classification_report(binary=True)
            report_gen.confusion_matrix(binary=False)
            report_gen.classification_report(binary=False)

            report_gen.roc(binary=False)

            report_gen.pr_curve(binary=False)

            # This will create approval and appendix pages
            report_gen.save_as_pdf("test_report.pdf")

    def test_create_complete_report(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        # although this is greyed out this is still a required import
        from providers.reportdata_provider import viNetResultDataProvider

        with viNetResultDataProvider(tag, config,
                                     classification_group=cgroup,
                                     procedure=sproc) as dp:
            # dp.set_limit(10000)
            dp.fetch_results()
            vinet_datum = dp.dataframe

            # DOnt actually need to set windfarm_name, as Vattenfall has one windfarm.
            vinet_datum.vi.set_attributes(customer=CustomerObjectMap["Vattenfall"](), windfarm_name='Gotland-Wind')

            ClassifierPerformanceReport.create_complete_report(vinet_datum, None, appendix=False)

    def test_frame_attribute_analytics(self):
        from providers.reportdata_provider import viNetResultDataProvider
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        cgroup = 'Vattenfall Proper 4 Class'
        with viNetResultDataProvider(tag, config, classification_group=cgroup, include_images=True) as dp:
            dp.set_limit(10000)
            dp.fetch_results()

            # Needed when dealing with images
            dp.map_results()
            vinet_datum = dp.dataframe
            vinet_datum.vi.set_attributes(customer=CustomerObjectMap["Vattenfall"](), windfarm_name='Gotland-Wind')

            cpplot = ClassifierPerformanceReport(vinet_datum)

            # plot basic plots. THe can also be done with their dedicated functions
            cpplot.plot_by_str('pie')
            cpplot.plot_by_str('cm')
            cpplot.plot_by_str('roc')
            cpplot.plot_by_str('pr')

            cpplot.plot_by_str('time')

            cpplot.plot_by_str('cor-brightness')
            cpplot.plot_by_str('cm', subdivision='brightness')
            cpplot.plot_by_str('cor-perceived_brightness')

            cpplot.plot_by_str('cor-variance')
            cpplot.plot_by_str('roc', subdivision='variance')
            cpplot.plot_by_str('cor-entropy')
            cpplot.plot_by_str('cor-noise')
            cpplot.plot_by_str('pr', subdivision='noise')

            cpplot.plot_by_str('cor-contrast')
            cpplot.plot_by_str('cm', subdivision='contrast')
            cpplot.plot_by_str('cor-fourier_spectrum_iq')

            cpplot.plot_by_str('cor-average_color-red')
            cpplot.plot_by_str('cor-average_color-blue')

            cpplot.plot_by_str('cor-contrast')
            cpplot.plot_by_str('cm', subdivision='contrast', subdiv_option='low')

            cpplot.save_as_pdf("test_report.pdf")

    def test_time_analytics(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        # although this is greyed out this is still a required import
        from providers.reportdata_provider import viNetResultDataProvider

        with viNetResultDataProvider(tag, config,
                                     classification_group=cgroup,
                                     procedure=sproc) as dp:
            # dp.set_limit(10000)
            dp.fetch_results()
            vinet_datum = dp.dataframe

            vinet_datum.vi.set_attributes(customer=CustomerObjectMap["Vattenfall"](), windfarm_name='Gotland-Wind')

            report_gen = ClassifierPerformanceReport(vinet_datum)

            report_gen.plot_by_str('cm')
            report_gen.plot_by_str('cm', binary=True)
            from core.plots.plot_data import MONTHS, SEASONS, SUN_TIMES
            for m in MONTHS:
                report_gen.plot_by_str('cm', False, 'month', m)

            for s in SEASONS:
                report_gen.plot_by_str('cm', False, 'season', s)

            for s in SUN_TIMES:
                report_gen.plot_by_str('cm', False, 'sun', s)

            report_gen.save_as_pdf('test_report.pdf')
