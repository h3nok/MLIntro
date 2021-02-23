
import os
from abc import ABC

from trdg.generators import GeneratorFromStrings

image_height = 64
image_width = 128
folder_label = 'series_year_train.txt'


class OCRFeature(ABC):
    def __init__(self, name, friendly_name):
        self._name = name
        self._friendly_name = friendly_name

    def friendly_name(self):
        return self._friendly_name

    def name(self):
        return self._name


class SeriesYear(OCRFeature):
    def __init__(self):
        super().__init__(name='Series Year', friendly_name='SY')
        self.corpus = []
        self.characters_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def allowed_years(self):
        with open(folder_label, 'r') as f:
            for line in f.readlines():
                if ';' in line:
                    _, txt = line.split(sep=';', maxsplit=1)
                    txt = txt.strip()
                    series_text_1 = "SERIES\n" + txt
                    self.corpus.append(series_text_1)
                    for ch in self.characters_list:
                        if ch in txt:
                            series_text_2 = "SERIES\n" + txt.replace(ch, '') + "\n" + ch
                            self.corpus.append(series_text_2)
                            
        self.corpus = list(set(self.corpus))

    def generate_images(self, output_dir='.'):
        if len(self.corpus) == 0:
            self.allowed_years()
        # font = ImageFont.truetype('fonts/Verdana.ttf', 24)
        font = 'fonts/CopperplateBold.ttf'
        # NOTE - this class call below is modified to generate multi-line text
        # which it doesn't do by default.
        generator = GeneratorFromStrings(self.corpus, fonts=[font],
                                         size=32, blur=0, random_blur=False,
                                         word_split=True, image_mode='L',
                                         text_color='Black', count=230,
                                         fit=True, background_type=3)
        counter = 0
        for img, lbl in generator:
            series = lbl.split()
            if len(series) == 3:
                series = series[1] + "_" + series[2]
            else:
                series = series[1]
            if not os.path.exists(os.path.join(output_dir, series)):
                os.makedirs(os.path.join(output_dir, series))
                counter = 0
            filename = os.path.join(output_dir, series, f"{series}.png")
            if os.path.exists(filename):
                filename = os.path.join(output_dir, series, f"{series}_{counter}.png")
                counter += 1

            img.save(filename)
