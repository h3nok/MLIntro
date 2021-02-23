from data_mapper_base import DatabaseObjectMapper
import database_interface as dbi
from common.db.stored_procedures import StoredProcedures as sp
from plots.distribution import Distribution as Dist
from matplotlib.backends.backend_pdf import PdfPages


class ClassificationCounts(object):
    def __init__(self, class_id, name, count):
        self.id = class_id
        self.name = name
        self.count = count


class ClassificationCountsMapper(DatabaseObjectMapper):
    """
    classification_counts database table mapper
    """
    def __init__(self):
        self.table = 'classification_count'
        self.schema = 'viclassify'

        self.server = dbi.PgsqlInterface()
        self.server.connect(name='classification_counts')

        self.counts = None
        self.classifications = None

    def fetch(self):
        self.counts = sp.classification_counts(self.server.connection)
        self.classifications = [ClassificationCounts(['name'], entry['id'], entry['count'])
                                for entry in self.counts.to_dict('r')]

    def classifications(self):
        if not self.classifications:
            self.fetch()
        return self.classifications


if __name__ == '__main__':
    mapper = ClassificationCountsMapper()
    mapper.fetch()

    print(mapper.counts)
