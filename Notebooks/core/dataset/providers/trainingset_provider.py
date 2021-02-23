from math import ceil
from common.profiler import Timer

from common.audit import endswith, is_file, path_exists
from logs import logger
from providers.provider_base import DataProviderBase

TIME_PER_1000_SEC = 172.08


class TrainingSetProvider(DataProviderBase):
    __query__ = None
    __server__ = None

    def __init__(self, query=None, server=None, batch_size=None, limit=None):
        """
        Creates a server side cursor for the supplied query as way to fetch batches
        of data from server without overwhelming local memory

        @type batch_size: int
        @param query: a sql script file or a query string enclosed in quotes
        @param server:
        """
        super().__init__()
        self._logger = logger.Logger(self.__name__, console=True).configure()
        assert query
        self.__query = None
        self.__batch_size = batch_size
        self.__limit = limit
        self._number_of_batches = None

        if self.__limit:
            self._number_of_batches = ceil(self.__limit/self.__batch_size)

        if is_file(query) or endswith(query, '.sql'):
            assert path_exists(query)
            self.__query = open(query).read()
        else:
            self.__query = query

        if self.__limit:
            self.__query += f"\n LIMIT {self.__limit}"

        self._validate()

        if not server:
            # Create named cursor to keep data on server side
            self.__server = self.database_hook()
            self.__server.connect(name='TrainingSetProvider')

        self.__server.cursor.itersize = self.__batch_size
        self._logger.debug("TrainingSetProvider initialized, query committed")

    def __enter__(self, query=None, query_file=None, server=None):
        """
        Context manager (entry)
        @param query:
        @param query_file:
        @param server:
        @return:
        """
        return self

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def query(self):
        return self.__query

    @property
    def number_of_batches(self):
        if self._number_of_batches:
            return self._number_of_batches-1
        return 1

    def _validate(self):
        assert self.__query, "Must supply sql query string to DatasetProvider"
        pass

    @Timer.function_timer
    def execute_query(self):
        self.__server.execute(self.__query)

    def analyze_query(self, query):
        """

        @param query:
        @return:
        """
        self._logger.info("Analyzing query ... \n {}".format(query))
        if not self.__limit:
            # calculate number of records
            cursor = self.__server.new_cursor(name="Benchmark Query")
            cursor.execute(query)
            records = cursor.fetchall()[0][0]
            self._number_of_batches = ceil(int(records)/self.__batch_size)
            cursor.close()
        else:
            records = self.__limit

        self._logger.debug("Number of records: {}, batches: {}".format(records, self._number_of_batches))

        def __seconds_to_hours(seconds):
            if seconds < 3600:
                return f'{round(seconds, 2)} secs'

            h = int(seconds // 3600)
            m = seconds % 3600 // 60
            s = seconds % 3600 % 60

            return '{:02d} hrs :{} mins :{} sec'.format(h, round(m, 2), round(s, 2))

        estimated_time = __seconds_to_hours((records/1000) * TIME_PER_1000_SEC)
        self._logger.debug("Estimated time to completion: {}".format(estimated_time))

        return records

    def get_batch(self):
        self._logger.info("Fetching samples, batch size: {}".format(self.__batch_size))
        return self.__server.fetch_many(self.__batch_size)

    def __exit__(self, query, query_file, server):
        """
            Clean up
        @param query:
        @param query_file:
        @param server:
        @return:
        """
        self.__server.close()
        del self.__query


