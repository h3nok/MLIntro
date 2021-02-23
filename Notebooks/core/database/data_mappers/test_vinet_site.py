from unittest import TestCase
from vinet_site import viNetSite


class TestviNetSite(TestCase):
    name = "viNetSite-Test-3"
    site = viNetSite(name)

    def test__str__(self):
        print(self.site)
