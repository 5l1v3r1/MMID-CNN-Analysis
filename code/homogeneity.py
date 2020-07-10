import os
import numpy as np
import csv
from sklearn.cluster import k_means
from sklearn.metrics.cluster import homogeneity_score
from dictionary import get_foreign_to_foreign_dictionary, get_translation_dictionary
from utils import *
import sys
import PATHS

# path to save the results to
OUTPUT_FILE = PATHS.SCORE_RESULTS_FOLDER + "/Homogeneity/{}_to_{}_homogeneity_scores.tsv"

langcode2name = get_language_code_mapping()

def save_scores(h_scores, language_code1, language_code2, sort=True):
    """ Helper function to save scores """
    file_name = OUTPUT_FILE.format(language_code1, language_code2)

    if h_scores is not None:
        if sort:
            h_scores = h_scores[h_scores[:, 2].astype(float).argsort()]

        np.savetxt(file_name, h_scores, delimiter='\t',
                    fmt='%s', encoding='utf-8')


def run_k_means(language_code1, language_code2, n_jobs=None, save_every=500, print_every=100, overwrite=True):
    """ Runs sklearn k-means algorithm on both language's matrices. The labels are removed from
    the matrices, then the matrices are concatenated. The homogeneity score is then calculated and
    written to a file called {language_code1}_to_{language_code2}_homogeneity_scores.tsv. Additionally,
    if a dictionary with mappings between the two languages doesn't exist one is created using
    google translate api.
    :param language_code1: ISO 639-1 language code for the first language
    :param language_code2: ISO 639-1 language code for the second language
    :param n_jobs: indicates how many cores k-means should use, see sklearn docs
    :param save_every: how often to update the homogeneity scores in # iterations
    :param print_every: how often to print status in # iterations
    :param overwrite: whether to overwrite the scores if a file already exists
    """
    # In case the user passes in same lang twice for the language code
    if language_code1 == language_code2:
        raise ValueError("Cannot provide translation")

    file_path = OUTPUT_FILE.format(language_code1, language_code2)
    if os.path.isfile(file_path) and not overwrite:
        print("File for {} to {} already exists and will not be overwritten".format(language_code1, language_code2))
        return

    # In case the user passes in 'en' for the language code
    if language_code1 == "en":
        translation_d = get_translation_dictionary(language_code2, inverted=True) # en to foreign
        cnn_index1 = get_cnn_index("en", language_code2) # en with foreign lang package
        cnn_index2 = get_cnn_index(language_code2, language_code2) # foreign with foreign lang package
    elif language_code2 == "en":
        translation_d = get_translation_dictionary(language_code1) # foreign to en
        cnn_index1 = get_cnn_index(language_code1, language_code1) # foreign with foreign lang package
        cnn_index2 = get_cnn_index("en", language_code1) # en with foreign lang package
    else:
        translation_d = get_foreign_to_foreign_dictionary(language_code1, language_code2) # foreign to foreign
        cnn_index1 = get_cnn_index(language_code1, language_code1) # foreign with foreign package
        cnn_index2 = get_cnn_index(language_code2, language_code2) # foreign with foreign package

    num_words = len(translation_d)

    h_score_list = np.zeros((num_words, 3), dtype=object)

    start = time.time()
 
    for i, lang1_word in enumerate(translation_d.keys()):
        if i % print_every == 0 and i != 0:
            print("{} to {} Iteration: {:6} Completed: {:6.2f}%% Elapsed Time: {} Time Remaining: {}".format(language_code1,
                                                                                                    language_code2,
                                                                                                    i,
                                                                                                    i/num_words * 100,
                                                                                                    timeSince(
                                                                                                        start),
                                                                                                    timeRemaining(start, i, num_words))
                  )

        if i % save_every == 0 and i != 0:
            save_scores(h_score_list, language_code1, language_code2)

        lang2_word = translation_d[lang1_word]

        if lang1_word in [None, ""] or lang2_word in [None, ""]:
            h_score_list[i, :] = np.array(
                    [lang1_word, lang2_word, None])
            continue

        # If it is a phrase and not in cnn index break it down
        if lang1_word not in cnn_index1 or lang2_word not in cnn_index2:

            lang1_word_list = lang1_word.split()
            lang2_word_list = lang2_word.split()

            lang1_word_list.reverse()
            lang2_word_list.reverse()

            for word in lang1_word_list:
                if word in cnn_index1:
                    lang1_word = word
                    break

            for word in lang2_word_list:
                if word in cnn_index2:
                    lang2_word = word
                    break

        # Check if broken down word failed or succeeded or skip
        # if original word in cnn index exists
        if lang1_word not in cnn_index1 or lang2_word not in cnn_index2:
            h_score_list[i, :] = np.array(
                [lang1_word, lang2_word, "Nan"])
            continue

        lang1_file_number = cnn_index1[lang1_word]
        lang2_file_number = cnn_index2[lang2_word]

        if language_code1 == "en":
            lang1_matrix = get_matrix(lang1_file_number, "en", language_code2) # english with foreign package
            lang2_matrix = get_matrix(lang2_file_number, language_code2, language_code2) # foreign from foreign package
        elif language_code2 == "en":
            lang1_matrix = get_matrix(lang1_file_number, language_code1, language_code1) # foreign from foreign package
            lang2_matrix = get_matrix(lang2_file_number, "en", language_code1) # english from foreign package
        else:
            lang1_matrix = get_matrix(lang1_file_number, language_code1, language_code1)
            lang2_matrix = get_matrix(lang2_file_number, language_code2, language_code2) 

        if lang1_matrix is None or lang2_matrix is None:
            h_score_list[i, :] = np.array(
                [lang1_word, lang2_word, None])
            continue

        m1, n1 = lang1_matrix.shape
        m2, n2 = lang2_matrix.shape

        if n1 != 4096 or n2 != 4096:
            h_score_list[i, :] = np.array(
                [lang1_word, lang2_word, None])
            continue


        true_labels = np.concatenate((np.zeros((1, m1)), np.ones((1, m2))), axis=1)
        data = np.concatenate((lang1_matrix, lang2_matrix), axis=0)
        centroid, pred_labels, _ = k_means(data, n_clusters=2)
        h_score = homogeneity_score(true_labels.flatten(), pred_labels.flatten())
        h_score_list[i, :] = np.array([lang1_word, lang2_word, h_score])

    save_scores(h_score_list, language_code1, language_code2)


def main():
    if len(sys.argv) == 3:
        lc1 = sys.argv[1].strip().lower()
        lc2 = sys.argv[2].strip().lower()
        if len(lc1) != 2 or len(lc2) != 2:
            raise ValueError("Please provide ISO 639-1 codes")
        run_k_means(lc1, lc2)
    elif len(sys.argv) == 2:
        lc = sys.argv[1].strip().lower()
        run_k_means(lc, 'en')
    elif len(sys.argv) == 4:
        lc1 = sys.argv[1].strip().lower()
        lc2 = sys.argv[2].strip().lower()
        overwrite = sys.argv[3].strip().lower()
        if len(lc1) != 2 or len(lc2) != 2:
            raise ValueError("Please provide ISO 639-1 codes")

        if overwrite == "y":
            run_k_means(lc1, lc2, overwrite=True)
        else:
            run_k_means(lc1, lc2)
    else:
        raise ValueError(
            "Incorrect Number of additional arguments passed, was expecting 1, 2, or 3, got {}".format(len(sys.argv) - 1))

if __name__ == '__main__':
    main()
