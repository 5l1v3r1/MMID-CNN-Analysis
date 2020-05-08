from utils import *
from dictionary import check_dictionary

LANGS = ("en", "fr", "ar", "az", "es", "id", "de", "tr", "hi", "it", "vi", "th", "cy")

for lc1 in LANGS:
    for lc2 in LANGS:
        print(lc1, lc2, check_dictionary(lc1, lc2))
