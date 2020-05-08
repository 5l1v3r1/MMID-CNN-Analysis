import time
import math
import csv
import pandas as pd
import os
import pickle
import PATHS

# Path to the a pkl file that contains a matrix for a language
MATRIX_PATH = PATHS.LANGUAGE_PACKAGES_PATH + "/{}/{}-features/{}.pkl"

INDEX_PATH = PATHS.LANGUAGE_PACKAGES_PATH + "/{}/{}_path_index.tsv"

COSINE_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Median-Max-Cosine-Similarity/{}_med_max_cosine.tsv"

HOMOGENEITY_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Homogeneity/{}_to_{}_homogeneity_scores.tsv"

def get_language_code_mapping():
    """ Creates dict that returns language from ISO 639-1 code key """
    path = PATHS.LANGUAGE_CODES
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[1]: row[2] for row in r}
    return d

langcode2name = get_language_code_mapping()

def get_matrix(file_number, language_code, lang_package):
    """ Retrieves matrix from pkl """
    lang_package = lang_package.lower()
    lang = langcode2name[language_code]
    lang = lang.strip().lower()

    file_path = MATRIX_PATH.format(lang_package, lang, file_number)

    if not os.path.isfile(file_path):
        file_path = MATRIX_PATH.format(lang_package, lang.title(), file_number)

    if not os.path.isfile(file_path):
        return None

    with open(file_path, 'rb') as fid:
        obj = pickle.Unpickler(fid, encoding='latin-1').load()

    return obj.toarray()

def get_cnn_index(language_code, lang_package):
    """ Gets mapping between language's words and file numbers """
    name = langcode2name[language_code].lower()
    lang_package = lang_package.lower()

    file_path = INDEX_PATH.format(lang_package, name)

    if not os.path.isfile(file_path):
        file_path = INDEX_PATH.format(lang_package, name.title())

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "Couldn't find index path for {} in package {}".format(language_code, lang_package))

    with open(file_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]: row[1] for row in r}

    return d


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%2dm %2ds' % (m, s)

def timeRemaining(since, i, num_iterations):
    if i == 0: i +=1

    now = time.time()
    s = now - since
    proportion = num_iterations/i
    total_time = (s * proportion)
    s = total_time - s
    m = math.floor(s / 60)
    s -= m * 60
    return '%2dm %2ds' % (m, s)

def writeDicttoTSV(d, filename):
    with open(filename, 'w+', encoding='utf-8') as f:
        for key, item in d.items():
            f.write("{}\t{}\n".format(key, item))

def readDictfromTSV(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]: float(row[1]) for row in r}
    return d

def readTSV(path, data_col_name):
    column_names = ["foreign word", "english word"] + data_col_name

    with open(path, 'r', encoding='utf-8') as f:
        df = pd.read_csv(f, delimiter='\t', names=column_names)
    return df.replace('None', None)

def print_hist2d(h):
    data = h[0]
    xedge = h[1]
    yedge = h[2]

    for i, row in enumerate(h[0]):
        print("{:5.0f} | ".format(yedge[i]), end="")
        for item in row:
            print("{:5.0f} ".format(item), end="")
        print()
    print(" "* 8, end="")
    for item in xedge[1:]: print("{:5.2f} ".format(item), end="")
    print("\n")

def read_cosine_TSV(language_code):
    path = COSINE_PATH.format(language_code)
    if os.path.isfile(path):
        with open(path, "r", encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            d = {row[0]: row[1] for row in r}
        return d
    else:
        # print("File with the name {} does not exist.".format(path))
        return None

def read_homogeneity_TSV(language_code1, language_code2):
    path = HOMOGENEITY_PATH.format(language_code1, language_code2)
    column_names = ["lang1_word", "lang2_word", "h_score", "english_word"]
    if os.path.isfile(path):
        with open(path, "r", encoding='utf-8') as f:
            df = pd.read_csv(f, delimiter='\t', names=column_names)
        return df.replace("None", None)
    else:
        # print("File with the name {} does not exist.".format(path))
        return None
