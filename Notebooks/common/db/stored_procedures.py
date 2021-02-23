import time

from pandas.io import sql as sqlio


class StoredProcedures:

    """
        Maps stored procedures to python strings
    @todo
        Move sql transactions to dbi
        Refactor, remove class and import functions by name
    """
    @staticmethod
    def dataset_tags(connection):
        """
        List all dataset tags
        @param connection:
        @return:
        """
        assert connection
        query = """select * from tags()"""
        result = sqlio.read_sql_query(query, connection)
        return [item.strip() for item in result['name']], [tag_id.strip() for tag_id in result['tag_id']]

    @staticmethod
    def vinet_configs(connection):
        """
        List all existing vinet Configurations
        @param connection:
        @return:
        """
        assert connection
        query = """select * from configs()"""
        return [item.strip() for item in sqlio.read_sql_query(query, connection)['name']]

    @staticmethod
    def get_random_frame(connection):
        """

        @param connection:
        @return:
        """
        assert connection
        query = """select * from get_random_frame()"""
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def get_frames(category, limit, connection):
        assert connection
        if not limit:
            limit = 1
        query = f"""select * from public.get_frames(\'{category}\', {limit})"""
        print(query)
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def get_classification_groups(connection):
        """

        @param connection:
        @return:
        """
        assert connection
        query = """select * from vinet.network_classification_group"""
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def get_network_truth_mapping(connection, grouping):
        """

        @param connection:
        @param grouping:
        @return:
        """
        assert connection
        assert grouping
        query = """select * from get_network_truth_mapping('{}')""".format(grouping)
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def insert_classification_group(connection, group_name, comment):
        """
        Adds a new classification group name to database

        @param connection:
        @param group_name:
        @param comment:
        @return:
        """
        assert connection
        assert group_name
        assert comment

        query = """call add_new_classification_group(\'{}\', \'{}\')""".format(group_name, comment)
        try:
            sqlio.execute(query, connection)
        except Exception as e:
            print(e)

    @staticmethod
    def insert_new_category(connection, category, protected):
        """

        @param connection:
        @param category:
        @param protected:
        @return:
        """
        query = """call public.add_new_classification(\'{}\', {})""".format(category, protected)
        try:
            sqlio.execute(query, connection)
        except Exception as e:
            print(e)

    @staticmethod
    def vinet_classification_groups(connection):
        """

        @param connection:
        @return: a table with headers classification_group_id, name, comment
        """
        assert connection

        query = """select * from vinet.network_classification_group"""
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def classification_labels(connection):
        """

        @param connection:
        @return:
        """
        query = """select * from source.classifications"""
        classifications = sqlio.read_sql_query(query, connection)

        return [(name, protected, idd) for name, protected, idd in zip(classifications['name'],
                                                                       classifications['protected'],
                                                                       classifications['id'])]

    @staticmethod
    def map_category_to_class_group(connection, class_group, category_id, group_class_id):
        """

        @param connection:
        @param class_group:
        @param category_id:
        @param group_class_id:
        @return:
        """
        query = """call add_category_to_classification_group(\'{}\', {}, {})""".format(class_group,
                                                                                       category_id,
                                                                                       group_class_id)
        try:
            sqlio.execute(query, connection)
        except Exception as e:
            print(e)

    @staticmethod
    def classification_group_info(classgroup_name, connection):
        """

        @param classgroup_name:
        @param connection:
        @return:
        """
        query = """select * from classification_group_info(\'{}\')""".format(classgroup_name)
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def site_data_distribution(site, connection, limit=None):
        """
        Retrieves available frames by site name

        @param limit:
        @param site:
        @param connection:
        @return:
        """

        query = """select truth, count from site_data_dist(\'{}\')""".format(site)
        if limit:
            query += "\nlimit {}".format(limit)
        print(f"Executing sproc, script: {query}")
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def vinet_sites(connection):
        """
        Retrieves names of available sites

        @param connection:
        @return:
        """
        query = """select name, site_id from source.sites"""
        print(f"Executing sproc, script: {query}")
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def classification_counts(connection):
        query = """SELECT id, "name", count FROM viclassify.classification_counts;"""
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def get_species_metadata(site, species, connection, **kwargs):
        query = """select * from vinet.get_species_metadata_from_site(\'{}\', \'{}\')""".format(site, species)
        print(f"Executing sproc, script: {query}")
        return sqlio.read_sql_query(query, connection)

    @staticmethod
    def add_new_tag(name: str, comment: str, tag_type: int, connection):
        query = """call public.add_new_tag(\'{}\', \'{}\', \'{}\')""".format(name, comment, tag_type)
        print(f"Executing sproc, script: {query}")
        try:
            sqlio.execute(query, connection)
        except Exception as e:
            print(e)

    @staticmethod
    def add_new_site(name: str, connection):
        query = """call public.add_new_site(\'{}\')""".format(name)
        print(f"Executing sproc, script: {query}")
        try:
            sqlio.execute(query, connection)
            time.sleep(3)
        except Exception as e:
            print(e)
