import core.database.database_interface as dbi
from abc import ABC, abstractmethod
from logs import logger as L
from common.db.stored_procedures import StoredProcedures as sp
from providers.provider_base import DataProviderBase


class viNetResultDataProvider(DataProviderBase):
    """
    __variable__ - enforced
    __variable - required
    _variable - optional

    """

    _server = None
    __tag = None
    __query_procedure = 'get_network_classification_results'
    _classification_group = None
    _result = None
    _logger = L.Logger('viNetResultDataProvider').configure()

    def __init__(self, tag, config, classification_group=None, procedure=None, limit=None, include_images=False):
        """
            Among other functionality this object Fetches viNet performance results
            as a dataframe
        @param tag:
        @param config:
        @param classification_group:
        @param procedure:
        @todo remove hardcoded stuff
        """
        self._query_cgroup = True
        if include_images:
            self.__query_procedure = 'get_network_classification_results_with_images'
            self._classification_group = classification_group
            self._query_cgroup = False
        if procedure:
            assert classification_group, "Must supply classification group."
            self.__query_procedure = procedure
            self._classification_group = classification_group
        assert self.__query_procedure
        self.__tag = tag
        self.__config = config
        self._limit = limit
        assert self.__tag and self.__config

        self._server = super().database_hook()
        self._server.connect()
        self._validate()

    def __enter__(self):
        """
            Context manager
        """
        return self

    def _validate(self) -> None:
        assert self.__query_procedure, "Must supply sql query string or stored procedure"
        tags, _ = sp.dataset_tags(self._server.connection)
        # assert self.__tag.lower().strip() in tags, "Supplied dataset tag doesn't exists, " \
        #                                            "tag: %s" % self.__tag

    def map_results(self):
        """
        Maps all of the ground truth elements to the grouping
        """
        assert self._classification_group and self._server
        assert 'groundtruth' in self._result.columns
        grouping = sp.get_network_truth_mapping(self._server.database_hook, self._classification_group)
        for i in range(len(self._result['groundtruth'])):
            self._result['groundtruth'][i] = self.get_groundtruth_net_mapping(self._result['groundtruth'][i], grouping)

    def get_groundtruth_net_mapping(self, gt, grouping=None):
        """ Given a value from the ground truth this will map it was learned as in the network"""
        assert gt
        if grouping is None:
            grouping = sp.get_network_truth_mapping(self._server.database_hook, self._classification_group)
        gts = grouping['groundtruth']
        mapping = grouping['group_name']

        for i in range(len(gts)):
            if gt == gts[i]:
                return mapping[i]

    def fetch_results(self) -> None:
        """
        Executes stored procedure
        @return:
        """
        if self._result is None:
            try:
                if self._classification_group and self._query_cgroup:
                    self._result = self._server.read_sproc(sp=self.__query_procedure,
                                                           config=self.__config,
                                                           tag=self.__tag,
                                                           group=self._classification_group, limit=self._limit)
                else:
                    self._result = self._server.read_sproc(sp=self.__query_procedure,
                                                           tag=self.__tag,
                                                           config=self.__config, limit=self._limit)
                if len(self._result.index) < 2:
                    raise RuntimeError("Stored procedure return empty "
                                       "dataframe, sproc: {}".format(self.__query_procedure))
            except Exception as e:
                self._logger.fatal(str(e))

    def set_limit(self, limit):
        self._limit = limit

    def to_csv(self, file):
        self._result.to_csv(file, index=False)

    def to_excel(self, file):
        self._result.to_excel(file, index=False)

    @property
    def dataframe(self) -> object:
        if self._result is None or len(self._result) == 0:
            self.fetch_results()
        self._result.vi.set_attributes(None, config=self.__config, tag=self.__tag)

        return self._result

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def __name__(self):
        return self.__class__.__name__

    def __exit__(self, tag, config, classification_group=None, procedure=None):
        del self.__tag
        del self.__config
        if self._classification_group:
            del self._classification_group
        del self.__query_procedure
        del self._result
        self._server.close()

