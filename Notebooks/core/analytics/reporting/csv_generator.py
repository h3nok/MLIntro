import os
from pandas import read_csv

class CSVGenerator:
    _file_base = None
    _file_writer = None
    _entries_str = 'entries'

    _data = []

    def __init__(self, file, allow_append=True, entries_str='entries'):
        """

        :param file: Must be .csv
        :param allow_append: Set to False if you want to overwrite any existing files
        :param entries_str: the entry value of the top left box
        """
        assert os.path.splitext(file)[1].lower() == '.csv'

        self._file_base = file
        assert ',' not in entries_str

        self._entries_str = entries_str

        if not allow_append:
            self._data = []
        else:
            self._data = self.read_file_data()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def dataframe(self):
        return read_csv(self._file_base)

    def close(self):
        """

        :return:
        """
        # self._file_writer.writelines(map(lambda s: s + '\n', self._data))
        # self._file_writer.close()

    def read_file_data(self):
        """
        reads data from file to an array

        Not intended for use of reading CSV data

        :return: Data: list of lines from file
        """
        if not os.path.exists(self._file_base):
            self._data = []
        else:
            with open(self._file_base, 'r') as f:
                content = f.readlines()
            self._data = [x.strip() for x in content]
        return self._data

    @staticmethod
    def __to_line(line_data):
        line = ''
        for part in line_data:
            part_str = str(part)
            if ',' in part_str:
                part_str = '"' + part_str + '"'
            line += part_str.replace('\n', '\\n') + ','
        return line.rstrip(',')

    @staticmethod
    def __to_line_data(line):
        return line.split(',')

    def write_line(self, entry: str, entry_data: dict):
        """
        Writes a single line dynamically; adds new columns if needed.

        Will write data to file.

        :param entry: name
        :param entry_data: dict of the data where the key is the column and the value is the entry
        :return:
        """
        new_line_data = [entry]

        if len(self._data) == 0:
            header_data = [self._entries_str]
            for data_entry in entry_data:
                header_data.append(data_entry)
            self._data = [self.__to_line(header_data)]

        for data_entry in self.__to_line_data(self._data[0])[1:]:
            if data_entry not in entry_data:
                new_line_data.append('')
            else:
                new_line_data.append(entry_data[data_entry])
                entry_data.pop(data_entry)

        for data_entry in entry_data:
            header_data = self.__to_line_data(self._data[0])
            header_data.append(data_entry)
            self._data[0] = self.__to_line(header_data)

            new_line_data.append(entry_data[data_entry])

        new_line = self.__to_line(new_line_data)
        self._data.append(new_line)

        with open(self._file_base, 'w') as self._file_writer:
            self._file_writer.writelines(map(lambda s: s + '\n', self._data))
