import csv
import PATHS

PATH = PATHS.SCORE_RESULTS_FOLDER + "/English-Median/{}_median_cosine.tsv"

LANGS = ["fr", "ar", "az", "es", "id",
    "de", "tr", "hi", "it", "vi", "th", "cy"]

d = {}

for lang in LANGS:
    filename = PATH.format("en_"+str(lang))
    with open(filename, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if row[0] not in d:
                d[row[0]] = (row[1], row[2])

output = PATH.format("en")
with open(output, 'w+', encoding='utf-8') as f:
    for key, item in d.items():
        f.write("{}\t{}\t{}\n".format(key, item[0], item[1]))
