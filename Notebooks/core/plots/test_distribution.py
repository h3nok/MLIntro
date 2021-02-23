from unittest import TestCase
from plots.distribution import Distribution as Dist
import matplotlib.pyplot as plt


class TestDistribution(TestCase):
    data = {'Unknown': 2336838, 'White-Tailed-Eagle': 359439}
    dist = Dist(data=data)

    def test_Pie(self):
        self.dist.pie._classified_frames_distribution()
        plt.show()

    def test_Hist(self):
        pass

