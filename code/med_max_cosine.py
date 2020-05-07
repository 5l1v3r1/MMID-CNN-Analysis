import os
import sys
import pickle
import numpy as np
import csv
from dictionary import get_foreign_to_foreign_dictionary, read_raw_dictionary
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import PATHS

# Score File path
RESULTS_PATH_SINGLE = PATHS.SCORE_RESULTS_FOLDER + "/Median-Max-Cosine-Similarity/{}_med_max_cosine.tsv"

# Path to Dictionaries
RAW_DICTIONARY_PATH = PATHS.RAW_DICTIONARY_PATH + "/dict."
PROCESSED_DICTIONARY_PATH = PATHS.DICTIONARY_PATH "/processed_dict."

# Path to the CNN folder with all language packages
LANG_PACKAGE_PATH = PATHS.LANGUAGE_PACKAGES_PATH

langcode2name = get_language_code_mapping()

def get_matrix(file_number, language_code, lang_package):
    lang_package = lang_package.lower()
    lang = langcode2name[language_code]
    lang = lang.strip().lower()

    if lang == 'english':
        file_path = LANG_PACKAGE_PATH + lang_package + \
            "/english-features/" + str(file_number) + ".pkl"
    else:
        file_path = LANG_PACKAGE_PATH + lang_package + "/" + \
            lang.title() + "-features/" + str(file_number) + ".pkl"

        if not os.path.isfile(file_path):
            file_path = LANG_PACKAGE_PATH + lang_package + "/" + \
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


def calc_med_max_cosine_similarity(language_code, print_every=100, save_every=500, overwrite = False):
    """ Given a ISO 639-1 language_code, calculate the median max cosine_similarity
    for every word in its raw dictionary.
    :param language_code: ISO 639-1 language_code
    :param save_every: how often to update the homogeneity scores in # iterations
    :param print_every: how often to print status in # iterations
    :param overwrite:whether to overwrite the scores if a file already exists
    """
    lang_package = langcode2name[language_code].lower()

    d = read_raw_dictionary(language_code)
    cnn_index = get_cnn_index(language_code, lang_package)

    num_words = len(d)

    dist_list = np.zeros((num_words, 3), dtype=object)

    file_name = RESULTS_PATH_SINGLE.format(language_code)

    start = time.time()

    print("STARTED COSINE SIMILIARITY CALCULATION FOR {}".format(lang_package.upper()))

    for i, word in enumerate(d.keys()):
        if i % save_every == 0 and i != 0:
            np.savetxt(file_name, dist_list, delimiter='\t',
                       fmt='%s', encoding='utf-8')

        if i % print_every == 0 and i != 0:
            print("{} Iteration: {:6} Completed: {:6.2f}%% Elapsed Time: {} Time Remaining: {}".format(language_code,
                                                                                                    i,
                                                                                                    i/num_words * 100,
                                                                                                    timeSince(start),
                                                                                                    timeRemaining(start, i, num_words))
                  )

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
        lc = sys.argv[1].strip().lower()
        ow = sys.argv[2].strip().lower()
        if ow == "y":
            calc_med_max_cosine_similarity(lc1, overwrite=True)
        else:
            calc_med_max_cosine_similarity(lc)
    elif len(sys.argv) == 2:
        lc = sys.argv[1].strip().lower()
        calc_med_max_cosine_similarity(lc)
    else:
        raise ValueError(
            "Incorrect Number of additonal arguments passed, was expecting 1 or 2, got {}".format(len(sys.argv) - 1))


if __name__ == '__main__':
    main()
