from common.db.stored_procedures import StoredProcedures as sp
from database_interface import PgsqlInterface
from data_mapper_base import DatabaseObjectMapper, MappedDatabaseObject


class viNetTrainingFrameLabel(MappedDatabaseObject):
    def __init__(self, category, protected, class_id=None):
        """
            Abstract database object representing a label or classification of a training example
        """

        self.name = category
        self.protected = protected
        self.id = class_id

    def __eq__(self, other):
        if self.name == other.name and self.id == other.id \
                and self.protected == other.protected:
            return True
        return False

    def __str__(self, ):
        return "__viNetFrameLabel(name: {}, id: {} protected: " \
               "{})__".format(self.name, self.id, self.protected)


class viNetTrainingFrameLabelMapper(DatabaseObjectMapper):

    """ A data mapper object that maps a pixel frame label (classification) table to a viNetTrainingFrameLabel
        in memory object
    """

    def __init__(self, frame_label: viNetTrainingFrameLabel):
        super().__init__()
        self.frame_label = frame_label
        self.name = frame_label.name
        self.protected = frame_label.protected
        self.connection = self.database_hook()
        self.id = frame_label.id

    def __enter__(self):
        return self

    def _exists(self):
        for c in self.get_persisted_data():
            if c == self.frame_label:
                return True

        return False

    def get_persisted_data(self):
        existing_classifications = sp.classification_labels(self.connection)
        vinet_labels = []
        for name, protected, class_id in existing_classifications:
            vinet_labels.append(viNetTrainingFrameLabel(name, protected, class_id))

        return vinet_labels

    def insert(self):
        if self._exists():
            print("Error: Supplied category \'{}\' already exists!".format(self.name))
            return False

        return sp.insert_new_category(self.connection, self.name, self.protected)

    def __exit__(self, exception_type, exception_value, traceback):
        del self.connection

