from customer.vattenfall import Vattenfall
from customer.e3 import E3
from customer.gwa import GWA
from customer.nextera import NextEra
from customer.totw import TOTW
from customer.avangrid import Avangrid


# if you want to build a vattenfall object you can simply do a lookup based on a key (customer name)
# Make it case insensitive if possible.

# Example: Customer['Vattenfall']()

CustomerObjectMap = {
    'Vattenfall': Vattenfall,
    'E3': E3,
    'GWA': GWA,
    'Goldwind': GWA,
    'NextEra': NextEra,
    "TOTW": TOTW,
    'Avangrid': Avangrid,
    'Clearway': None,
    'Longroad': None
}


class CustomerFactory(object):
    """
    TODO - dynamically construct a new customer objects
    """
    pass
