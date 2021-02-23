from unittest import TestCase
from vattenfall import Vattenfall
from avangrid import Avangrid
from e3 import E3
from totw import TOTW
from nextera import NextEra
from gwa import GWA
from customer.customers import CustomerObjectMap


class TestCustomer(TestCase):
    def testConstructors(self):
        vattenfall = Vattenfall()
        assert vattenfall

    def testVattenfall(self):
        customer = Vattenfall()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

    def testAvangrid(self):
        customer = Avangrid()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

        for species in customer.deployed_network_classifications:
            assert species.name in ['Eagle-Or-Kite',
                                    'Buzzard', 'Gull',
                                    'Other-Avian-Gotland']

    def testE3(self):
        customer = E3()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

        for species in customer.deployed_network_classifications:
            assert species.name in ['Eagle-Or-Kite',
                                    'Buzzard', 'Gull',
                                    'Other-Avian-Gotland']

    def testTOTW(self):
        customer = TOTW()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

        for species in customer.deployed_network_classifications:
            assert species.name in ['Eagle-Or-Kite',
                                    'Buzzard', 'Gull',
                                    'Other-Avian-Gotland']

    def testNextEra(self):
        customer = NextEra()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

        for species in customer.deployed_network_classifications:
            assert species.name in ['Eagle-Or-Kite',
                                    'Buzzard', 'Gull',
                                    'Other-Avian-Gotland']

    def testGWA(self):
        customer = GWA()
        assert customer.application.value
        assert customer.geo_location.address
        assert customer.geo_location.coordinates
        assert customer.windfarms
        assert customer.protected_species

        for species in customer.deployed_network_classifications:
            assert species.name in ['Eagle-Or-Kite',
                                    'Buzzard', 'Gull',
                                    'Other-Avian-Gotland']

    def test_customers(self):
        cust = CustomerObjectMap["Vattenfall"]()
        assert cust.name
        cust = CustomerObjectMap["E3"]()
        assert cust.name
        cust = CustomerObjectMap["GWA"]()
        assert cust.name
        cust = CustomerObjectMap["NextEra"]()
        assert cust.name
        cust = CustomerObjectMap["TOTW"]()
        assert cust.name
