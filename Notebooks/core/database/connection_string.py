import os
from configparser import ConfigParser


class ConnectionStringFactory:

    @staticmethod
    def url_from_ini(ini_file='database.ini', section='postgresql') -> dict:
        """

        @param section:
        @param ini_file: ini file name containing database connection info
        @return: connection string
        """

        parser = ConfigParser()

        if not os.path.exists(ini_file):
            raise RuntimeError("database ini file not found, filename: {}".format(ini_file))

        parser.read(ini_file)
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception("Section {0} not found in the {1} file".format(section, ini_file))

        return db

    @staticmethod
    def create_pgsql_connection_string(host='12.1.1.91',
                                       port='5432',
                                       database='viNet',
                                       user=None,
                                       password=None) -> dict:
        """

        @param host:
        @param port:
        @param database:
        @param user:
        @param password:
        @return:
        """

        db = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

        return db

