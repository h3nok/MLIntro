from time import sleep

from tabulate import tabulate

from common.db.stored_procedures import StoredProcedures as sp
from data_mapper_base import DatabaseObjectMapper, MappedDatabaseObject
from vinet_frame_label import viNetTrainingFrameLabel, viNetTrainingFrameLabelMapper


class viNetClassGroup(MappedDatabaseObject):
    """
    A classification group - abstract database object that consists a
    mapping of a one or more labels
    """

    def __init__(self, name, comment=None, group_id=None):
        # TODO - fetch all classifications \ labels mapped to the this class group.
        # use class_group_info stored procedure
        self.name = name
        self.comment = comment
        self.id = group_id
        self.mapped_labels = []

    def __str__(self):
        return f"""
        viNetClassGroup: 
            Name: {self.name}
            Id:{self.id} 
            Comments: {self.comment}
        """

    def __eq__(self, other):
        if self.name == other.name and self.id == other.id:
            return True
        return False


class viNetClassGroupMapper(DatabaseObjectMapper):
    """
        An interface to classification group (database table)
    """

    table = 'network_classification_group'

    def __init__(self, classification_group: viNetClassGroup = None):
        super().__init__()
        self.classification_group = classification_group
        self.connection = self.database_hook()
        self.existing_class_groups = [class_group for
                                      class_group in self.get_persisted_data()['name']]
        self.known_labels = [viNetTrainingFrameLabel(c[0], c[1], c[2])
                             for c in sp.classification_labels(self.connection)]

    def __enter__(self) -> object:
        return self

    def get_persisted_data(self):
        return sp.vinet_classification_groups(self.connection)

    def exists(self):
        if self.classification_group.name in self.existing_class_groups:
            print("Class group \'{}\' already exists".format(self.classification_group.name))
            return True
        else:
            return False

    def insert(self, class_group: viNetClassGroup = None):
        """
        Inserts a classification group, ClassGroup(name, comment, id) to the database.

        @param class_group:
        @return:
        """
        try:
            if not self.classification_group:
                assert class_group
                self.classification_group = class_group

            if not self.exists():
                sp.insert_classification_group(self.connection, class_group.name, class_group.comment)
                print(f"Successfully added class group, name: {class_group.name}")
                return True
            return True
        finally:
            return False

    def map_classifications(self, labels_list: list,
                            group_class: viNetTrainingFrameLabel,
                            classification_group: viNetClassGroup = None):
        """
        Map a list of labels to a group label for viNet development

        ex. [Golden-Eagle, Bald-Eagle ] ==> Eagle

        @param classification_group: The classification group, a viNetClassGroup object
        @param labels_list: a list of classifications to be mapped to a one group class
        @param group_class: a grouping label of the list of classifications
        @return:
        """
        if classification_group:
            self.classification_group = classification_group

        assert labels_list
        assert group_class

        # check if the classification group entry exists
        # insert it to database if it doesn't exist
        if not self.exists():
            print("Group label \'{}\' not found in database. "
                  "Inserting ...".format(self.classification_group.name))
            self.insert()
            sleep(5)
            # self.existing_classifications = set([viNetTrainingFrameLabel(c[0], c[1], c[2])
            #                                      for c in sp.classifications(self.connection)])

        # Make sure all the  to-be-grouped classification exist in the database.
        # Insert all of them if not in the db
        for label in labels_list:
            if label not in self.known_labels:
                with viNetTrainingFrameLabelMapper(label) as lm:
                    lm.insert()

        sleep(3)
        # Need class id to be present in the database to be mapped, update the existing class container
        # after insert
        self.known_labels = {c[0]: c[2]
                             for c in sp.classification_labels(self.connection)}
        # Get the class group id after inserted to db
        group_class.id = self.known_labels[group_class.name]

        for label in labels_list:
            label_name = label.name
            if label_name in self.known_labels.keys():
                label_id = self.known_labels[label_name]

                print(f'Mapping {label_name}:{label_id} <===> '
                      f'{self.classification_group.name}:{self.classification_group.id}')

                sp.map_category_to_class_group(self.connection,
                                               self.classification_group.name,
                                               label_id, group_class.id)
        sleep(5)
        print("Successfully updated classification group")
        print(tabulate(sp.classification_group_info(self.classification_group.name, self.connection),
                       headers='keys', tablefmt='psql'))

        return True

    def __exit__(self, exception_type, exception_value, traceback):
        self.connection.close()
