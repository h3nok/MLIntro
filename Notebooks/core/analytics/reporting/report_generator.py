from core.analytics.reporting.pdf_generator import PDFGenerator
from providers.reportdata_provider import viNetResultDataProvider
from core.datatype.vinet_dataframe_plotter import DataFramePlotter
from customer.customers import CustomerObjectMap


class Report:
    """
        reporter base class
    """
    _data = None
    __plotter = None
    __pdf_generator = None
    _binary_allowed = False
    _pdf_gen_provided = False
    _plot_dpi = None

    __pdf_title = None

    def __init__(self, data, title='Classification Report', pdf_gen=None, quality="normal", colors=None):
        """
        this is a base class for both performance and correlation
        @param data: dataframe object to create plots from
        @param pdf_gen: an outside pdf generator can be provided.
        @param quality: can be string "low", "normal", "high" or int for the dpi value
        @param title: acts as an override for __pdf_title
        """
        self._data = data.vi
        self.__pdf_title = title
        if pdf_gen:
            self.__pdf_generator = pdf_gen
            self._pdf_gen_provided = True
        else:
            self.__pdf_generator = PDFGenerator()
            self.__pdf_generator.create_page('title', config=self._data.config, tag=self._data.tag,
                                             title=self.__pdf_title)
        if len(self._data.binary_classifications) == 2:
            self._binary_allowed = True

        self.__plotter = DataFramePlotter(self._data, colors=colors)

        if isinstance(quality, int):
            self._plot_dpi = quality
        elif isinstance(quality, str):
            if quality.lower() == "low":
                self._plot_dpi = 400
            elif quality.lower() == "high":
                self._plot_dpi = 800
            elif quality.lower() == "ultra":
                self._plot_dpi = 1200
            else:  # "normal" case
                self._plot_dpi = 600
        else:
            raise TypeError("quality param must be string or int")

    @property
    def pdf_generator(self):
        """ This is so other things can be added to the pdf. should only be used if a pdf_gen was not provided. """
        return self.__pdf_generator

    def plot_by_str(self, plot_type, binary=False, subdivision=None, subdiv_option=None):
        """
        given input string will create the plot
        @param plot_type: string. ex: cm, roc
        @param binary: binary plot
        @param subdivision:
        @param subdiv_option:
        """
        if binary and not self._binary_allowed:
            print("Binary plot could not be created, no protected category, skipped.")
            return
        fig = self.__plotter.plot(plot_type, binary, subdivision, subdiv_option)
        self.__pdf_generator.create_page('plot', plot_fig=fig, plot_dpi=self._plot_dpi)

    def save_as_pdf(self, save_loc=None, approval=True, appendix=True, open_pdf=False):
        """
        Used to save pdf. If a pdf generator was not provided this will also add the approval and appendix pages
        otherwise if the generator was provided that must be done outside of this class. But the pdf can still be saved
        @param open_pdf:
        @param save_loc: location of pdf to be saved. When non the config will be the file
        @param approval: should the approval page be added
        @param appendix: should the appendix be added
        """
        if approval:
            self.__pdf_generator.create_page('approval', config=self._data.config)
        if appendix:
            self.__pdf_generator.create_page('appendix')
        if save_loc is not None:
            self.__pdf_generator.save_pdf(save_loc, open_pdf=open_pdf)
        else:
            self.__pdf_generator.save_pdf(f"{self._data.config}.pdf", open_pdf=open_pdf)

    @staticmethod
    def create_complete_report(data, save_loc, appendix=False):
        raise NotImplementedError()


def config_tag_to_dataframe(tag, config, classification_group=None, get_images=False, customer=None):
    """
    A method to automatically collect necessary data. I might add this the reporting class if needed
    @param tag:
    @param config:
    @param classification_group:
    @param get_images:
    @param customer:
    """
    sproc = 'get_network_classification_results_on_grouped_frames'
    if get_images:
        dp = viNetResultDataProvider(tag, config, classification_group=classification_group, include_images=True)
    else:
        dp = viNetResultDataProvider(tag, config, classification_group=classification_group, procedure=sproc)

    dp.fetch_results()
    if get_images:
        dp.map_results()

    vinet_datum = dp.dataframe
    if customer is not None:
        vinet_datum.vi.set_attributes(customer=CustomerObjectMap[customer]())

    return vinet_datum
