import os
import sys
import pickle
import numpy as np
import csv
from dictionary import get_foreign_to_foreign_dictionary, read_raw_dictionary
from main import get_cnn_index, get_language_code_mapping
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import PATHS

# Results Folder path
RESULTS_PATH = PATHS.SCORE_RESULTS_FOLDER + "/{}_to_{}_median_cosine.tsv"
RESULTS_PATH_SINGLE = PATHS.SCORE_RESULTS_FOLDER + "/{}_median_cosine.tsv"

# path to the spanish to english word translation
RAW_DICTIONARY_PATH = PATHS.RAW_DICTIONARY_PATH + "/dict."
PROCESSED_DICTIONARY_PATH = PATHS.DICTIONARY_PATH "/processed_dict."

# path to the CNN folder
LANG_PACKAGE_PATH = PATHS.LANGUAGE_PACKAGES_PATH

langcode2name = get_language_code_mapping()


def get_matrix(file_number, language_code, lang_package1):
    lang_package1 = lang_package1.lower()
    lang = langcode2name[language_code]
    lang = lang.strip().lower()

    if lang == 'english':
        file_path = LANG_PACKAGE_PATH + lang_package1 + \
            "/english-features/" + str(file_number) + ".pkl"
    else:
        file_path = LANG_PACKAGE_PATH + lang_package1 + "/" + \
            lang.title() + "-features/" + str(file_number) + ".pkl"

        if not os.path.isfile(file_path):
            file_path = LANG_PACKAGE_PATH + lang_package1 + "/" + \
                lang.lower() + "-features/" + str(file_number) + ".pkl"

        if not os.path.isfile(file_path):
            return None

    with open(file_path, 'rb') as fid:
        obj = pickle.Unpickler(fid, encoding='latin-1').load()

    return obj.toarray()


def get_cnn_index(language_code, lang_package):
    name = langcode2name[language_code].lower()
    lang_package = lang_package.lower()

    if name == "english":
        tsv_path = LANG_PACKAGE_PATH + lang_package + "/english_path_index.tsv"

        if not os.path.isfile(tsv_path):
            raise FileNotFoundError(
                "Couldn't find index path for english in package {}".format(lang_package))
    else:
        tsv_path = LANG_PACKAGE_PATH + lang_package + \
            "/" + name.title() + "_path_index.tsv"

        if not os.path.isfile(tsv_path):
            tsv_path = LANG_PACKAGE_PATH + lang_package + \
                "/" + name.lower() + "_path_index.tsv"

        if not os.path.isfile(tsv_path):
            raise FileNotFoundError(
                "Couldn't find index path for {} in package {}".format(language_code, lang_package))

    with open(tsv_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]: row[1] for row in r}

    return d

def median_max_cosine(matrix):
    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, -1)
    max_vals = np.max(similarities, axis=0)
    median_max = np.median(max_vals)
    return [median_max, matrix.shape[0]]


def calc_cosine_similiarity_one(language_code):
    lang_package = langcode2name[language_code].lower()

    d = read_raw_dictionary(language_code)
    cnn_index = get_cnn_index(language_code, lang_package)

    num_words = len(d)

    dist_list = np.zeros((num_words, 3), dtype=object)

    file_name = RESULTS_PATH_SINGLE.format(language_code)

    start = time.time()

    print("STARTED COSINE SIMILIARITY CALCULATION FOR {}".format(lang_package.upper()))

    for i, word in enumerate(d.keys()):
        if i % 100 == 0 and i != 0:
            np.savetxt(file_name, dist_list, delimiter='\t',
                       fmt='%s', encoding='utf-8')
        scores = [None] * 2

        if word not in [None, ""]:
            if word not in cnn_index:
                dist_list[i, :] = np.array([word]+["Nan"]*2, dtype=object)
                continue

            path = cnn_index[word]
            mat = get_matrix(path, language_code, lang_package)

            if mat is None:
                dist_list[i, :] = np.array([word]+["Nan"]*2, dtype=object)
                continue

            scores = median_max_cosine(mat)

        dist_list[i, :] = np.array([word] + scores, dtype=object)

    temp = dist_list[dist_list[:, 1].astype(float).argsort()]
    np.savetxt(file_name, temp, delimiter='\t', fmt='%s', encoding='utf-8')
    print("FINISHED COSINE SIMILIARITY CALCULATION FOR {}".format(lang_package.upper()))

def main():
    if len(sys.argv) == 3:
        lc1 = sys.argv[1].strip().lower()
        lc2 = sys.argv[2].strip().lower()
        calc_cosine_similiarity(lc1, lc2)
    elif len(sys.argv) == 2:
        lc = sys.argv[1].strip().lower()
        calc_cosine_similiarity_one(lc)
    else:
        raise ValueError(
            "Incorrect Number of arguments passed, was expecting 2 or 3, got {}".format(len(sys.argv)))


if __name__ == '__main__':
    main()
