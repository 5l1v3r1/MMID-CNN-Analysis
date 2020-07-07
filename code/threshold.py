import os
import pickle
import numpy as np
import csv
import sys
from dictionary import *
from utils import *
from prettytable import PrettyTable
import PATHS

RESULTS_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Threshold/{}"


def threshold(homogeneity_threshold, cosine_threshold, lc1, lc2):
    cosine_d1 = read_cosine_TSV(lc1)
    cosine_d2 = read_cosine_TSV(lc2)

    df = read_homogeneity_TSV(lc1, lc2)

    rows = np.zeros((3, 4), dtype=int).tolist()

    row_names = ["High Homogeneity", "Low Homogeniety", "Total"]
    column_names = ["Both High Score",
                    "One High Score", "Both Low Score", "Total"]

    t = PrettyTable()
    t.field_names = [""] + column_names

    for data_struct in (cosine_d1, cosine_d2, df):
        if data_struct is None:
            t = PrettyTable()
            t.field_names = [""] + column_names

            for i, row in enumerate(rows):
                t.add_row([row_names[i]] + row)
            return t

    del df["english_word"]
    df["lang1_median_max_cosine"] = df["lang1_word"].map(cosine_d1)
    df["lang2_median_max_cosine"] = df["lang2_word"].map(cosine_d2)

    df = df.replace("Nan", np.nan).replace("None", None).dropna()

    for col in df.columns[2:]:
        df[col] = df[col].astype(float)
    df = df[df["lang1_median_max_cosine"] != -1]
    df = df[df["lang2_median_max_cosine"] != -1]

    high_h_df = df[df["h_score"] > homogeneity_threshold]
    low_h_df = df[df["h_score"] <= homogeneity_threshold]

    for i, data in enumerate([high_h_df, low_h_df, df]):
        temp = data[data["lang1_median_max_cosine"] > cosine_threshold]
        rows[i][0] = len(
            temp[temp["lang2_median_max_cosine"] > cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"]
                               <= cosine_threshold])

        temp = data[data["lang1_median_max_cosine"] <= cosine_threshold]
        rows[i][2] = len(
            temp[temp["lang2_median_max_cosine"] <= cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"]
                               > cosine_threshold])

        rows[i][3] = len(data)

    row_arr = np.array(rows)

    column_sum = np.sum(row_arr[:-1, :], axis=0)
    column_total = row_arr[-1, :]
    row_sum = np.sum(row_arr[:, :-1], axis=1)
    row_total = row_arr[:, -1]

    col_sum_sanity = True
    row_sum_sanity = True

    if (column_sum != column_total).any():
        col_sum_sanity = False

    if (row_sum != row_total).any():
        row_sum_sanity = False

    for i, row in enumerate(rows):
        t.add_row([row_names[i]] + row)

    return t


def get_nans(language_code1, language_code2):
    result = check_dictionary(language_code1, language_code2)
    result = list(result)
    result[-1] = "{:2.2f}%%".format(result[-1])
    t = PrettyTable()
    t.field_names = ["Usable Words", "Unusable Words", "Total", "% Usable"]
    t.add_row(result)
    return t


def threshold_all(homogeneity_threshold, cosine_threshold, langs, file_name):
    big_table = PrettyTable()
    big_table.field_names = ["Translation", "Result", "Num Nan"]

    s = "{} to {}"

    for lang1 in langs:
        for lang2 in langs:
            if lang1 != lang2:
                row = [s.format(lang1, lang2)]
                sub_t = threshold(homogeneity_threshold,
                                  cosine_threshold, lang1, lang2)
                nans = get_nans(lang1, lang2)
                row.append(sub_t)
                row.append(nans)
                big_table.add_row(row)
    with open(file_name, "w+") as f:
        f.write(big_table.get_string())


def get_ratio(homogeneity_threshold, cosine_threshold, lc1, lc2):
    cosine_d1 = read_cosine_TSV(lc1)
    cosine_d2 = read_cosine_TSV(lc2)

    df = read_homogeneity_TSV(lc1, lc2)

    rows = np.zeros((3, 4), dtype=int).tolist()

    row_names = ["High Homogeneity", "Low Homogeniety", "Total"]
    column_names = ["Both High Score",
                    "One High Score", "Both Low Score", "Total"]

    t = PrettyTable()
    t.field_names = [""] + column_names

    for data_struct in (cosine_d1, cosine_d2, df):
        if data_struct is None:
            return -1

    del df["english_word"]
    df["lang1_median_max_cosine"] = df["lang1_word"].map(cosine_d1)
    df["lang2_median_max_cosine"] = df["lang2_word"].map(cosine_d2)

    df = df.replace("Nan", np.nan).replace("None", None).dropna()

    for col in df.columns[2:]:
        df[col] = df[col].astype(float)
    df = df[df["lang1_median_max_cosine"] != -1]
    df = df[df["lang2_median_max_cosine"] != -1]

    high_h_df = df[df["h_score"] > homogeneity_threshold]
    low_h_df = df[df["h_score"] <= homogeneity_threshold]

    for i, data in enumerate([high_h_df, low_h_df, df]):
        temp = data[data["lang1_median_max_cosine"] > cosine_threshold]
        rows[i][0] = len(
            temp[temp["lang2_median_max_cosine"] > cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"]
                               <= cosine_threshold])

        temp = data[data["lang1_median_max_cosine"] <= cosine_threshold]
        rows[i][2] = len(
            temp[temp["lang2_median_max_cosine"] <= cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"]
                               > cosine_threshold])

        rows[i][3] = len(data)
    if len(df) != 0:
        ratio = rows[2][0]/rows[-1][-1]
        return ratio
    else:
        return -1


def rank_all(homogeneity_threshold, cosine_threshold, langs):
   for lang in langs:
       worst, best, ratio_list = rank(
           homogeneity_threshold, cosine_threshold, lang, langs)

       print(lang.upper())
       print("Worst", worst, "Best", best)
       for label, ratio in ratio_list:
            print("{} : {}, ".format(label, ratio), end="")
       print("\n\n")


def rank(homogeneity_threshold, cosine_threshold, lc, langs):
    ratio_list = []
    best = -10
    worst = 10
    for lang in langs:
        ratio = get_ratio(homogeneity_threshold, cosine_threshold, lc, lang)

        ratio_list.append(("{} to {}".format(lc, lang), ratio))

        if ratio < worst and ratio != -1:
            worst = ratio
            worst_s = "{} to {}".format(lc, lang)

        if ratio > best and ratio != -1:
            best = ratio
            best_s = "{} to {}".format(lc, lang)

    return worst_s, best_s, ratio_list


def main():
    LANGS = ["en", "fr", "ar", "az", "es", "id", "de", "tr", "hi", "it", "vi", "th", "cy"]
    obj = Threshold(0.15, 0.45, LANGS)
    obj.print_ranks()
    obj.print_thresholds()

if __name__ == "__main__":
    main()
