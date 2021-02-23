# # This whole file is not needed any more but I will keep it for reference
#
# from enum import Enum
#
#
# class viNetCustomer(Enum):
#     TOTW = 'Duke-TOTW'
#     E3 = 'E3'
#     Avangrid = 'Avangrid'
#     Longroad = 'Longroad'
#     NorthAmerica = 'Avangrid, TOTW, NextEra, LongRoad '
#     Vattenfall = 'Vattenfall-Gotland'
#     Goldwind = 'Goldwind-CattleHill'
#
#
# class TOTW(object):
#     GOLDEN_EAGLE = 'Golden-Eagle'
#     BALD_EAGLE = 'Bald-Eagle'
#     TURBINE_BLADE = 'Turbine-Blade'
#     TURBINE_ANTENNA = 'Turbine-Antenna'
#     TURBINE_HUB = 'Turbine-Hub'
#     RAVEN = 'Raven'
#     HAWK = 'Hawk'
#     TURBINE_OTHER = 'Turbine-Other'
#     CAL_TARGET = 'Calibration-Target'
#     TURKEY_VULTURE = 'Turkey-Vulture'
#     YOUNG_BALD = 'Young-Bald-Eagle'
#     YOUNG_GOLDEN = 'Young-Golden-Eagle'
#     OTHER_AVIAN = 'Other-Avian'
#     TURBINE_GENERAL = 'Turbine-General'
#     TURBINE_DARK = 'Turbine-Dark'
#     TURBINE_VENT = 'Turbine-Vent'
#     CONDOR = 'Condor'
#
#     TURBINE_ARTIFACTS = [TURBINE_GENERAL, TURBINE_ANTENNA, TURBINE_HUB,
#                          TURBINE_BLADE, TURBINE_OTHER, CAL_TARGET, TURBINE_VENT]
#     PROTECTED_SPECIES = [GOLDEN_EAGLE, BALD_EAGLE, YOUNG_BALD, YOUNG_GOLDEN]
#     PROTECTED_SUBSPECIES = [YOUNG_BALD, YOUNG_GOLDEN]
#
#     CLASSIFICATIONS = [GOLDEN_EAGLE, BALD_EAGLE, TURKEY_VULTURE, HAWK, RAVEN, TURBINE_GENERAL]
#     v24_CLASSIFICATIONS = [GOLDEN_EAGLE, BALD_EAGLE, TURKEY_VULTURE, HAWK]
#     v25_CLASSIFICATIONS = [GOLDEN_EAGLE, BALD_EAGLE, TURKEY_VULTURE, HAWK, RAVEN]
#     v26_CLASSIFICATIONS = [GOLDEN_EAGLE, BALD_EAGLE, TURKEY_VULTURE, HAWK, TURBINE_GENERAL]
#
#
# class Vattenfall(object):
#     WTE = 'White-Tailed-Eagle'
#     GOLDEN_EAGLE = 'Golden-Eagle'
#     RAVEN = 'Raven'
#     RoB_KITE = 'Red-Or-Black-Kite'
#     BUZZARD = 'Buzzard'
#     COMMON_BUZZARD = 'Common-Buzzard'
#     OTHER_AVIAN = 'Other-Avian-Gotland'
#     # EAGLE = 'Eagle'
#     COMMON_RAVEN = 'Common-Raven'
#     EAGLE = 'Golden-Or-WT-Eagle'
#     Eagle_Or_Kite = 'Eagle-Or-Kite'
#     Gull = 'Gull'
#
#     # v27_CLASSIFICATIONS = [GOLDEN_EAGLE, WTE, RoB_KITE, BUZZARD, OTHER_AVIAN]
#     v27_CLASSIFICATIONS = [BUZZARD, Eagle_Or_Kite, Gull, OTHER_AVIAN]
#     PROTECTED = [GOLDEN_EAGLE, WTE, Eagle_Or_Kite]
#
#
# class Avangrid_Classes(object):
#     GOLDEN_EAGLE = 'Golden-Eagle'
#     BALD_EAGLE = 'Bald-Eagle'
#     RAVEN = 'Raven'
#     HAWK = 'Hawk'
#     TURKEY_VULTURE = 'Turkey-Vulture'
#     YOUNG_BALD = 'Young-Bald-Eagle'
#     YOUNG_GOLDEN = 'Young-Golden-Eagle'
#
#     PRETECTED_SPECIES = [GOLDEN_EAGLE, BALD_EAGLE, YOUNG_GOLDEN, YOUNG_GOLDEN]
#     PROTECTED_SUBSPECIES = [YOUNG_BALD, YOUNG_GOLDEN]
#     AVANGRID_CLASSIFICATION_SPECIES = [GOLDEN_EAGLE, BALD_EAGLE, RAVEN, HAWK, TURKEY_VULTURE]
#
#
# class E3(object):
#     RED_KITE = 'Red-Kite'
#     HARRIER = 'Harrier'
#     WTE = 'White-Tailed-Eagle'
#     BUZZARD = 'Buzzard'
#     OTHER = "Other-Avian-DE"
#     BLACK_KITE = "Black-Kite"
#     KITE = "Red-Or-Black-Kite"
#
#     v22_CLASSIFICATION_SPECIES = {RED_KITE, KITE, WTE, BUZZARD, OTHER}
#     E3_PROTECTED_SPECIES = {RED_KITE, KITE, WTE}
#
#     BUZZARD_TYPES = ["Rough-Legged-Buzzard", "Buzzard-sp.",
#                      "Honey-Buzzard", "Eurasian-Buzzard"]
#     HARRIER_TYPES = ["Hen-Harrier", "Marsh-Harrier"]
#     KITE_TYPES = ['Red-Kite', 'Black-Kite']
#     EAGLE_TYPES = [WTE, 'Eagle']
#
#     def get_network_class(self, species):
#         species = species.lower()
#         if self.HARRIER.lower() in species:
#             return self.HARRIER
#         elif self.BUZZARD.lower() in species:
#             return self.BUZZARD
#         elif 'Eagle'.lower() in species:
#             return self.WTE
#         elif 'Kite'.lower() in species:
#             return self.RED_KITE
#         else:
#             return self.OTHER
#
#
# class Goldwind(object):
#     WEDGE_TAILED_EAGLE = 'Wedge-Tailed-Eagle'
#     HAWK_FALCON = 'Hawk-Falcon'
#     RAVEN = 'Raven'
#     Other_Avian = 'Other-Avian'
#
#
# class Customer(Enum):
#     e3 = 'E3'
#     totw = 'TOTW'
#     avangrid = 'Avangrid'
#     na = 'North America'
