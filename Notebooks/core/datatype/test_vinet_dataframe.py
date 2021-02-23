from unittest import TestCase

import matplotlib.pyplot as plt
from providers.reportdata_provider import viNetResultDataProvider
from customer.customers import Vattenfall


class TestviNetDataframe(TestCase):

    def test_plot(self):
        try:
            config = 'viNet_2.7_Vattenfall_4_class_6.3m'
            tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
            sproc = 'get_network_classification_results_on_grouped_frames'
            cgroup = 'Vattenfall Proper 4 Class'
            # First we must create our data provided
            with viNetResultDataProvider(tag, config, classification_group=cgroup, procedure=sproc) as dp:
                # then we fetch results with the data provider
                # number of rows to grab. comment tis line out for all rows
                dp.set_limit(10000)
                dp.fetch_results()

                # if include_images, a different procedure is used so we need to call our mapping function
                # dp.map_results()

                # Once data has been fetched we can get the pandas dataframe, this will automatically give us
                # the vinetdataframe class when ever we use .vi
                vinet_datum = dp.dataframe

                # set customer. This is important for getting the protected classes and the geographical locations
                # set windfarm_name. if no name is provided then it will automatically use the first windfarm in the arr
                vinet_datum.vi.set_attributes(customer=Vattenfall(), ocnfig=config, tag=tag)
                assert len(vinet_datum.vi.protected_classifications) == 1
                vinet_datum.vi.profile_report(save_loc="profile_report.html")

                # once customer is set and data is loaded we can create our plotter class to create plots
                from core.datatype.vinet_dataframe_plotter import DataFramePlotter
                plotter = DataFramePlotter(vinet_datum)

                # Plot a confusion matrix with plotter
                plotter.plot('cm')
                plt.show()

                # plot binary confusion matrix with plotter
                plotter.plot('cm', binary=True, subdivision="month", subdiv_option='January')
                plt.show()

            self.doCleanups()

        except Exception as e:
            print(e)
            raise RuntimeError("Unable to run TestviNetDataframe failed")
            self.fail()

    def test_customer(self):
        # self.fail()
        pass

    def test_columns(self):
        # self.fail()
        pass

    def test_head(self):
        # self.fail()
        pass

    def test_classifications(self):
        # self.fail()
        pass

    def test_set_attributes(self):
        pass

    def test_calc_metrics(self):
        config = 'viNet_2.7_Vattenfall_4_class_6.3m'
        tag = 'Vattenfall-Gotland Training V2 (4 class) - Validation'
        sproc = 'get_network_classification_results_on_grouped_frames'
        cgroup = 'Vattenfall Proper 4 Class'
        with viNetResultDataProvider(tag, config, classification_group=cgroup, procedure=sproc) as dp:
            # number of rows to grab. comment tis line out for all rows
            #dp.set_limit(10000)
            dp.fetch_results()
            vinet_datum = dp.dataframe

            # sets "protected" classes and binarizes data into Protected vs Other
            vinet_datum.vi.binarize(['Eagle-Or-Kite'])

            metrics = vinet_datum.vi.calc_metrics('all')

            assert 'accuracy' in metrics
            assert 'fnr' in metrics
            assert 'fpr' in metrics
            assert 'tnr' in metrics
            assert 'tpr' in metrics
