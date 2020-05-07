from utils import *

d = get_language_code_mapping()

langs = ("en", "fr", "ar", "az", "es", "id", "de", "tr", "hi", "it", "vi", "th", "cy")

for lang in langs:
    print("-", d[lang])
