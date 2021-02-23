from core.analytics.correlation import ClassifierCorrelationPlots
from customer.customers import CustomerObjectMap


class TestClassifierCorrelationPlots:
    def test_create_complete_report(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        # although this is greyed out this is still a required import
        from providers.reportdata_provider import viNetResultDataProvider
        with viNetResultDataProvider(tag, config, classification_group=cgroup, include_images=True) as dp:
            dp.set_limit(10000)
            dp.fetch_results()
            # Needed when dealing with images
            dp.map_results()

            vinet_datum = dp.dataframe
            # DOnt actually need to set windfarm_name, as Vattenfall has one windfarm.
            vinet_datum.vi.set_attributes(customer=CustomerObjectMap["Vattenfall"]())
            ClassifierCorrelationPlots.create_complete_report(vinet_datum, None, appendix=False)

    def test_create_fine_grained_report(self):
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

            ClassifierCorrelationPlots.create_fine_grained_report(vinet_datum, None, False)
