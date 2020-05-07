import time
import math
import csv
import pandas as pd
import os
import PATHS

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


def get_language_code_mapping():
    path = PATHS.LANGUAGE_CODES
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[1]: row[2] for row in r}
    return d

def read_cosine_TSV(language_code):
    PATH  = "/project/multilm/nikzad/Analyze-CNN-code/Score-Results/{}_median_cosine.tsv"
    path = PATH.format(language_code)
    if os.path.isfile(path):
        with open(path, "r", encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            d = {row[0]: row[1] for row in r}
        return d
    else:
        # print("File with the name {} does not exist.".format(path))
        return None

def read_homogeneity_TSV(language_code1, language_code2):
    PATH = "/project/multilm/nikzad/Analyze-CNN-code/Score-Results/{}_to_{}_homogeneity_scores.tsv"
    path = PATH.format(language_code1, language_code2)
    column_names = ["lang1_word", "lang2_word", "h_score", "english_word"]
    if os.path.isfile(path):
        with open(path, "r", encoding='utf-8') as f:
            df = pd.read_csv(f, delimiter='\t', names=column_names)
        return df.replace("None", None)
    else:
        # print("File with the name {} does not exist.".format(path))
        return None
