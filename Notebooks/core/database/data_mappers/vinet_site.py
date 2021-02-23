import calendar
import os

from pandas import DataFrame, read_csv

from common.db.stored_procedures import StoredProcedures as sp
from core.analytics.reporting.pdf_generator import PDFGenerator as PDF
from core.database.data_mappers.data_mapper_base import DatabaseObjectMapper, MappedDatabaseObject
from core.plots.confusion_matrix import ConfusionMatrix as CM
from core.plots.distribution import Distribution as Dist
from core.plots.plot_data import viNetValidationData
from logger import Logger


class viNetSite(MappedDatabaseObject, DatabaseObjectMapper):
    """
        A site where viNet is deployed or a potential site for viNet

        # TODO - this doesn't conform to data_mapper pattern. Refactor,
        # see DatasetTag or ClassificationGroup
        # TODO - remove hardcoded stuff

    """

    def __init__(self,
                 name,
                 site_id=None,
                 working_dir=None):

        assert name
        self._logger = Logger(module=self.__class__.__name__,
                              console=True).configure()
        self.working_dir = None
        self._datum = DataFrame()
        self._plots = []
        self._working_dir = working_dir
        self._known_sites = []  # sites that exist in the database

        self.connection = self.database_hook()

        self.classified_frames_cache = None
        self.category_metadata_cache = None
        self.name = name
        self.Id = None
        if site_id:
            self.Id = site_id

        self.get_persisted_data()
        if self.name in self._known_sites:
            self._logger.debug("Found site name in database")
        else:
            self.insert()
            while self.name not in self._known_sites:
                self.get_persisted_data()

        self.Id = self._known_sites[self.name]

        # self._classified_frames_distribution()

    def __str__(self):
        return f"""
        viNetSite:
            Name: {self.name}
            Id: {self.Id} """

    def __eq__(self, other):
        if self.Id is not None and other.Id is not None:
            return self.Id == other.Id
        else:
            return self.name == other.name

    def _fetch_labeled_frames_metadata(self):
        if self._datum.empty:
            self._datum = sp.site_data_distribution(self.name, self.connection, limit=10)

        return self._datum

    def has_data(self):
        return not self._datum.empty

    def to_excel(self, filepath=None):
        if not filepath:
            filepath = os.path.join(self._working_dir, self.name, self.name + '.xlsx')

        self._datum.to_excel(filepath)

    def to_csv(self, filepath=None):
        if not filepath:
            filepath = os.path.join(self._working_dir, self.name, self.name + '.csv')

        self._datum.to_csv(filepath)

    def _classified_frames_distribution(self, highlight=None, plot=False):
        if self._datum.empty and not self.classified_frames_cache:
            self._fetch_labeled_frames_metadata()
        else:
            # load cached data
            self._datum = read_csv(self.classified_frames_cache)
        if plot:
            data = self.to_dict()
            data = {d['truth']: d['count'] for d in data}
            labels = list(data.keys())
            counts = list(data.values())
            title = "{} Frame Distribution, Total Frames: {}".format(self.name, sum(counts))

            with Dist(counts, labels, title) as dist:
                hist = dist.hist.plot(exceptions=highlight)
                self._plots.append(hist)

    def to_dict(self):
        return self._datum.to_dict('r')

    def _species_metadata(self,
                          category,
                          groundtruth=None,
                          classgroup=None):
        assert category
        if self._datum.empty and not self.category_metadata_cache:
            self._datum = sp.get_species_metadata(self.name, category, self.connection)
        else:
            self._datum = read_csv(self.category_metadata_cache)
        self._datum['inspection_time'] = self._datum['inspection_time'].astype('datetime64')

        # plot distance distribution
        distance = list(self._datum['distance'])
        dist = Dist(distance)
        title = "{} Distance Distribution, # frames: {}".format(category, len(distance))
        hist = dist.hist.plot_number_distribution(filter=(0, 5000), title=title)
        self._plots.append(hist)

        # plot monthly frame distributions
        months = self._datum.groupby(self._datum['inspection_time'].dt.month).count()
        labels = [calendar.month_name[x] for x in months['distance'].keys()]
        counts = list(months['distance'])
        title = "{} Monthly Distribution".format(category)

        with Dist(counts, labels, title) as dist:
            monthly_dist = dist.hist.plot()
            self._plots.append(monthly_dist)

        # plot confusion matrix
        if groundtruth:
            preds = []
            confidence = []
            for cat, conf in zip(self._datum['nn_category'], self._datum['nn_confidence']):
                if cat in classgroup:
                    preds.append(cat)
                    confidence.append(conf)
            gts = [groundtruth] * len(preds)
            cm_data = viNetValidationData(gts, preds, confidence, list(set(preds)))
            cm = CM(cm_data)
            cm_fig = cm.plot_confusion_matrix(title="viNet 2.2. Confusion matrix of {}".format(category))
            self._plots.append(cm_fig)

    def generate_report(self,
                        wdr,
                        category=None,
                        groundtruth=None,
                        classgroup=None) -> None:
        """
        This function can be used to summarize attributes of existing data of a given species
        For instance the LSE of E3.

        @param wdr: reports etc save location
        @param category: the species (as it exists in the database - LSE)
        @param groundtruth: (its groundtruth, if different from category, LSE's groundtruth for instance is WTE)
        @param classgroup:
        @return:
        """

        if not wdr:
            filename = self.name + ".pdf"
        else:
            filename = os.path.join(wdr, self.name, self.name + ".pdf")
        self.classified_frames_cache = os.path.join(wdr,
                                                    self.name,
                                                    self.name + "_classified_frames.csv")
        self._classified_frames_distribution(highlight=[category], plot=True)
        if not os.path.exists(self.classified_frames_cache):
            self.to_csv(filepath=self.classified_frames_cache)
        self._datum = DataFrame()
        if category:
            self.category_metadata_cache = os.path.join(wdr, self.name,
                                                        self.name + "_{}_metadata.csv".format(category))
            self._species_metadata(category=category, groundtruth=groundtruth, classgroup=classgroup)
            if not os.path.exists(self.category_metadata_cache):
                self.to_csv(filepath=self.category_metadata_cache)
        if len(self._plots) == 0:
            print("No plots found. Unable to generate report")
            return

        PDF().generate_from_plots(save_loc=filename,
                                  tag=category,
                                  config='Pre-viNet',
                                  plots=self._plots,
                                  title='viNet Dataset Report',
                                  appendix=False,
                                  approval=False)

    def get_persisted_data(self):
        """

        """
        sites = sp.vinet_sites(self.connection)
        site_names = sites['name']
        site_ids = sites['site_id']
        self._known_sites = {name: Id for name, Id in zip(site_names, site_ids)}

    def insert(self):
        """

        """
        sp.add_new_site(self.name, self.connection)
