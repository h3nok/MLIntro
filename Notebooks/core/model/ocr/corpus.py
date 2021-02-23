from dataset_utils import csv_to_dataframe
from features import OCRFeature


class Corpus:
    def __init__(self, dataframe):
        self._dataframe = dataframe

        self.columns = ["Series", "Series Years Allowed (Series)",
                        "Series Year Revision Allowed",
                        "Check Letter Characters Allowed (CL)",
                        "Quadrant Check Letter (QCL)", "Quadrant Number (Q#)",
                        "Face Plate Number (FP)"    "Back Plate Number (BP)"]
        self.series_year = set()

    @property
    def series_list(self):
        return self._dataframe[self.columns[0]]

    @property
    def series_year_list(self):
        return self._dataframe[self.columns[1]]

    def build_series_year_character_set(self, output_file):
        for sy in self.series_year_list:
            sy = sy.split(',')
            sy = set(sy)
            self.series_year.update(sy)
        with open(output_file, 'w+') as file:
            for series in self.series_year:
                series = series.strip()
                file.write(series + ".png;"+series+"\n")
        file.close()

csv_path = r"C:\Users\hghebrechristos\Desktop\viAI\Fed\Character_Recognition_Matrix.csv"
df = csv_to_dataframe(csv_path)
print(df.columns)

corpus = Corpus(df)
corpus.build_series_year_character_set('series_year_train.txt')
