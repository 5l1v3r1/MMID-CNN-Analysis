import os
import sys
import numpy as np
import csv
from dictionary import get_foreign_to_foreign_dictionary, read_raw_dictionary
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import PATHS

# Score File path
RESULTS_PATH_SINGLE = PATHS.SCORE_RESULTS_FOLDER + "/Median-Max-Cosine-Similarity/{}_med_max_cosine.tsv"

# ENGLISH SUB SCORE PATH
ENGLISH_MEDIAN_PATH = PATHS.SCORE_RESULTS_FOLDER + "/English-Median/en_{}_med_max_cosine.tsv"

# Path to Dictionaries
RAW_DICTIONARY_PATH = PATHS.RAW_DICTIONARY_PATH + "/dict."
PROCESSED_DICTIONARY_PATH = PATHS.DICTIONARY_PATH + "/processed_dict."

def median_max_cosine(matrix):
    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, -1)
    max_vals = np.max(similarities, axis=0)
    median_max = np.median(max_vals)
    return [median_max, matrix.shape[0]]


def calc_med_max_cosine_similarity(language_code, language_package=None, print_every=100, save_every=500, overwrite=False):
    """ Given a ISO 639-1 language_code, calculate the median max cosine_similarity
    for every word in its raw dictionary.
    :param language_code: ISO 639-1 language_code
    :param language_package: ISO 639-1 code for different language package. Only for english
    :param save_every: how often to update the homogeneity scores in # iterations
    :param print_every: how often to print status in # iterations
    :param overwrite:whether to overwrite the scores if a file already exists
    """
    if language_package is None:
        language_package = language_code
        d = read_raw_dictionary(language_code)
        file_name = RESULTS_PATH_SINGLE.format(language_code)
    elif language_code == "en":
        d = read_raw_dictionary(language_package, invert=True)
        file_name = ENGLISH_MEDIAN_PATH.format(language_package)
    else:
        raise AssertionError("Can only pass language_package with english language code")


    cnn_index = get_cnn_index(language_code, language_package)

    num_words = len(d)

    dist_list = np.zeros((num_words, 3), dtype=object)

    start = time.time()

    print("STARTED COSINE SIMILIARITY CALCULATION FOR {}".format(language_code.upper()))

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
                wordls = word.split()
                wordls.reverse()

                for w in wordls:
                    if w in cnn_index:
                        word = w
                        break
            
            if word not in cnn_index:
                dist_list[i, :] = np.array([word]+["Nan"]*2, dtype=object)


            path = cnn_index[word]
            mat = get_matrix(path, language_code, language_package)

            if mat is None or mat.shape[1] != 4096:
                dist_list[i, :] = np.array([word]+["Nan"]*2, dtype=object)
                continue

            scores = median_max_cosine(mat)

        dist_list[i, :] = np.array([word] + scores, dtype=object)

    temp = dist_list[dist_list[:, 1].astype(float).argsort()]
    np.savetxt(file_name, temp, delimiter='\t', fmt='%s', encoding='utf-8')
    print("FINISHED COSINE SIMILIARITY CALCULATION FOR {}".format(language_code.upper()))

def main():
    if len(sys.argv) == 3:
        lc = sys.argv[1].strip().lower()
        lp = sys.argv[2].strip().lower()
        calc_med_max_cosine_similarity(lc, language_package=lp)
    elif len(sys.argv) == 2:
        lc = sys.argv[1].strip().lower()
        calc_med_max_cosine_similarity(lc)
    else:
        raise ValueError(
            "Incorrect Number of additonal arguments passed, was expecting 1 or 2, got {}".format(len(sys.argv) - 1))


if __name__ == '__main__':
    main()
