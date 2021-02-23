from abc import ABC, abstractmethod

import pandas.io.sql as sqlio
import psycopg2
import psycopg2.extras

from common.profiler import Timer
from core.database.connection_string import ConnectionStringFactory
from logs import logger as L

_logger = L.Logger(__name__, console=True).configure()


class DatabaseInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def connection(self):
        pass


class PgsqlInterface(DatabaseInterface):
    """
    Database interface, psycopg2 wrapper
    """

    def __init__(self):
        self._connection = None
        self._params = None
        self._cursor = None
        self._named_cursor = None

    def __enter__(self):
        return self

    @staticmethod
    def connect_local(server='localhost', database='viNet-RnD', user='henok', pwd='vinet', port='5000'):
        try:
            _logger.debug("Connecting to local database, host: {}, database: {}, user: {}, "
                          "pwd: {}".format(server, database, user, pwd))
            connection = psycopg2.connect(user=user, password=pwd, host=server, port=port,
                                          database=database)
            return connection
        except (Exception, psycopg2.DatabaseError) as e:
            _logger.error(e)
            raise RuntimeError(e)

    def connect(self, ini=r'C:\ProgramData\viNet\config\database.ini', name=None, autocommit=False) -> object:
        """
        @todo - remove hardcoded config file
        """
        try:
            self._params = ConnectionStringFactory.url_from_ini(ini_file=ini)
            assert self._params is not None, "Empty database url_from_ini"
            _logger.info("Connecting to the PostgreSQL database...")
            self._connection = psycopg2.connect(**self._params)
            if autocommit:
                self._connection.set_session(autocommit=True)

        except (Exception, psycopg2.DatabaseError) as error:
            _logger.error(error)
            raise RuntimeError(error)
        finally:
            if self._connection is not None:
                _logger.info(
                    "Successfully connected to server. Creating server side cursor, name = {}".format(name))
                if name is not None:
                    self._cursor = self._connection.cursor(name=name, cursor_factory=psycopg2.extras.DictCursor)
                else:
                    self._cursor = self._connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

                self._cursor.itersize = 10000
                _logger.info('Done')
                return True

    @property
    def connection(self) -> object:
        """"
            Opens connection to database and returns a connection object.
        """
        if not self._connection:
            self.connect()
        return self._connection

    @property
    def cursor(self, name=None) -> psycopg2.extras.DictCursor:
        """
        Creates pgs transaction cursor. If name is provided, it creates a server side cursor
        @param name: cursor name
        @return:
        """
        if self._cursor is None:
            try:
                if name is not None:
                    self._cursor = self._connection.cursor(name=name,
                                                           cursor_factory=psycopg2.extras.DictCursor)
                else:
                    self._cursor = self._connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

                self._cursor.itersize = 10000

                if self._cursor is None:
                    _logger.error(
                        "Unable to retrieve database cursor, exiting ...")
                    raise ValueError("Unable to get_as_tfexample cursor from connection")
                else:
                    _logger.info(
                        "Successfully retrieved cursor, server: : {}".format(self._params['host']))
                    return self._cursor
            except (Exception, psycopg2.DatabaseError) as error:
                _logger.error(error)
        else:
            return self._cursor

    def new_cursor(self, name):
        """
            Creates a news named cursor
        @param name:
        @return:
        """
        return self._connection.cursor(name=name, cursor_factory=psycopg2.extras.DictCursor)

    def fetch_one(self) -> any:
        """
            Returns a single row from a cursor
        @return:
        """
        try:
            return self._cursor.fetchone()
        except (Exception, psycopg2.DatabaseError) as error:
            _logger.error(error)

    def fetch_all(self) -> any:
        """"
            Returns the entire table of a transaction
        """
        try:
            return self._cursor.fetchall()
        except (Exception, psycopg2.DatabaseError) as error:
            _logger.error(error)

    def fetch_many(self, batch_size=1000) -> any:
        """
            Returns a fixed size batch
        @param batch_size: number of rows to fetch
        @return:
        """
        try:
            results = self._cursor.fetchmany(batch_size)
            return results
        except (Exception, psycopg2.DatabaseError) as error:
            _logger.error(error)

    @Timer.function_timer
    def execute(self, query, limit=None, offset=None):
        """

        @param query:
        @param limit:
        @param offset:
        @return:
        """
        assert isinstance(query, str)

        if self._cursor is None:
            _logger.warn("Empty cursor, creating one...")
            self._cursor = self.cursor
        try:
            _logger.debug("Executing query, limit: {}, offset: {} ".format(limit, offset))
            if limit is None:
                _logger.debug(query)
                self._cursor.execute(query)
            elif limit and offset:
                query = "{}\n LIMIT {} \n OFFSET {}".format(query, limit, offset)
                _logger.debug(query)
                self._cursor.execute(query)

        except (Exception, psycopg2.DatabaseError) as e:
            _logger.error(e)

    @staticmethod
    def _construct_sproc_query_string(**kwargs):
        """
        First default argument must be sp or stored_procedure
        followed by any number of parameters (i.e. the number of parameters must match parameters of
        the stored procedure
        Example - sp=fetch_results, tag=this_tag, config='that_config' ...
        TODO - not generic enough. Requires knowledge of the stored procedure
        @param kwargs:
        @return:
        """
        query = str()
        comma_counter = 1
        limit = None
        comma_needed = len(kwargs.values()) - 2

        # construct sql query string ex. sp(param1, param2, param3);
        for key, value in kwargs.items():
            # first argument is the name of the stored procedure
            if key == 'sp' or key == 'stored_procedure':
                query += value + "("

            # the rest are parameters of the stored procedure
            else:
                # skip arguments having None value
                if value is None:
                    continue
                # limit is not the procedure argument,
                # applied after the select query is constructed
                if 'limit' in key:
                    limit = value
                    comma_needed -= 1
                    continue
                # wrap strings in single quotes (sql string args)
                if isinstance(value, str):
                    value = '\'' + value + '\''

                if comma_counter < comma_needed:
                    query += str(value) + ","
                    comma_counter += 1
                else:
                    query += str(value)
        query = 'select * from ' + query + ")"
        if limit and limit > 0:
            query += "\nlimit {};".format(limit)

        return query

    def read_sproc(self, **kwargs):
        """
        Executes read_sql_query to fetch data as a df. Use call_sproc if you do not need df
        @type kwargs: dict, coma separated arguments starting with stored procedure

        """

        query = self._construct_sproc_query_string(**kwargs)

        if self._cursor is None:
            _logger.warn("Empty cursor, creating one...")
            self._cursor = self.cursor
        try:
            _logger.info("Executing query, query: %s" % query)
            return sqlio.read_sql_query(query, self._connection)
        except (Exception, psycopg2.DatabaseError) as e:
            _logger.error(e)

    def close(self):
        self._cursor.close()
        self._connection.close()

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

