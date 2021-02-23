import os
from unittest import TestCase
from core.analytics.reporting.csv_generator import CSVGenerator


class TestCSVGenerator(TestCase):
    def test_write_line(self):
        file = 'test.csv'

        with CSVGenerator(file) as csvg:
            entry1 = 'test'
            entry1_data = dict()
            entry1_data['test1'] = 1
            entry1_data['test2'] = 1
            entry1_data['test3'] = 1

            entry2 = 'test2'
            entry2_data = dict()
            entry2_data['test1'] = 2  # give the dict entries some randomization
            entry2_data['test3'] = 2
            entry2_data['test2'] = 2

            entry3 = 'test3'
            entry3_data = dict()
            entry3_data['test4'] = 3
            entry3_data['test2'] = 3
            entry3_data['test1'] = 3
            entry3_data['test3'] = 3

            entry4 = 'test4'
            entry4_data = dict()
            entry4_data['test1'] = 4
            entry4_data['test4'] = 4
            entry4_data['test3'] = 4

            csvg.write_line(entry1, entry1_data)
            csvg.write_line(entry2, entry2_data)
            csvg.write_line(entry3, entry3_data)
            csvg.write_line(entry4, entry4_data)

        with open(file, 'r') as f:
            data = f.readlines()

        comp_data = \
            """ \
            entries,test1,test2,test3,test4
            test,1,1,1
            test2,2,2,2
            test3,3,3,3,3
            test4,4,,4,4
            """

        assert list(map(lambda l: l.rstrip('\n'), data)) == \
               list(map(lambda l: l.lstrip('\t').lstrip(' '), comp_data.split('\n')))[:-1]

        # clean up
        os.remove(file)
