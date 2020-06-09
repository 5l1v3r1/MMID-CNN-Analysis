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

class Threshold:
    def __init__(self, homogeneity_threshold, cosine_threshold, languages):
        self.h = homogeneity_threshold
        self.c = cosine_threshold
        self.ignore_less_than = 100
        self.langs = languages
        self.num2lang = {i: lang for i, lang in enumerate(languages)}
        self.lang2num = {lang: i for i, lang in enumerate(languages)}
        self.num_langs = len(languages)
        self.thresholds = np.zeros((len(languages), len(languages)), dtype=object)
        self.threshold_many()
        self.ratios = np.zeros((len(languages), len(languages)), dtype=object)
        self.get_ratios()
        self.ranks = self.rank_all()
        self.row_names = ["High Homogeneity", "Low Homogeniety", "Total"]
        self.column_names = ["Both High Score",
                        "One High Score", "Both Low Score", "Total"]

    def threshold(self, lc1, lc2):
        if lc1 == lc2:
            return

        cosine_d1 = read_cosine_TSV(lc1)
        cosine_d2 = read_cosine_TSV(lc2)

        df = read_homogeneity_TSV(lc1, lc2)

        rows = np.zeros((3, 4), dtype=int).tolist()

        del df["english_word"]
        df["lang1_median_max_cosine"] = df["lang1_word"].map(cosine_d1)
        df["lang2_median_max_cosine"] = df["lang2_word"].map(cosine_d2)

        df = df.replace("Nan", np.nan).replace("None", None).dropna()

        for col in df.columns[2:]:
            df[col] = df[col].astype(float)


        df = df[df["lang1_median_max_cosine"]!=-1]
        df = df[df["lang2_median_max_cosine"]!=-1]

        high_h_df = df[df["h_score"] > self.h]
        low_h_df = df[df["h_score"] <= self.h]

        for i, data in enumerate([high_h_df, low_h_df, df]):
            temp = data[data["lang1_median_max_cosine"] > self.c]
            rows[i][0] = len(temp[temp["lang2_median_max_cosine"] > self.c])
            rows[i][1] += len(temp[temp["lang2_median_max_cosine"] <= self.c])

            temp = data[data["lang1_median_max_cosine"] <= self.c]
            rows[i][2] = len(temp[temp["lang2_median_max_cosine"] <= self.c])
            rows[i][1] += len(temp[temp["lang2_median_max_cosine"] > self.c])

            rows[i][3] = len(data)

        row_arr = np.array(rows)

        return row_arr

    def sanity_check(row_arr):
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

        return col_sum_sanity and row_sum_sanity

    def threshold_many(self):
        for i, lc1 in enumerate(self.langs):
            for j, lc2 in enumerate(self.langs):
                if lc1 != lc2:
                    self.thresholds[i, j] = self.threshold(lc1, lc2)

    def get_ratio(self, row_arr):
        if type(row_arr) == int:
            return -1

        if row_arr[-1, -1] < self.ignore_less_than:
            return -1

        return row_arr[2, 0]/row_arr[-1, -1]

    def get_ratios(self):
        ratios = np.zeros((self.num_langs), dtype=object)
        for i in range(self.num_langs):
            for j in range(self.num_langs):
                self.ratios[i, j] = self.get_ratio(self.thresholds[i, j])

    def rank(self, index):
        ratio_list = self.ratios[index]

        best = -10
        best_index = -1
        worst = 10
        worst_index = -1

        for i in range(self.num_langs):
            ratio = ratio_list[i]
            if ratio != -1:
                if best < ratio:
                    best = ratio
                    best_index = i
                if worst > ratio:
                    worst = ratio
                    worst_index = i

        if best_index == -1 or best_index == -1:
            return "Nan", "Nan", "Nan", "Nan"
        best_lang = self.num2lang[best_index]
        worst_lang = self.num2lang[worst_index]

        return best, best_lang, worst, worst_lang

    def rank_all(self):
        rank_list = []
        for i in range(self.num_langs):
            rank_list += [self.rank(i)]

        return rank_list

    def print_ranks(self):
        t = PrettyTable()

        t.field_names = ["Language", "Most Similiar", "Least Similiar"]

        for i in range(self.num_langs):
            best, best_lang, worst, worst_lang = self.ranks[i]
            t.add_row([self.num2lang[i], best_lang, worst_lang])

        print(t)

    def get_single_threshold_table(self, i, j):
        row_arr = self.thresholds[i, j]

        t = PrettyTable()
        t.field_names = [""] + self.column_names

        for i, name in enumerate(self.row_names):
            t.add_row([name] + row_arr[i].tolist())

        return t

    def get_nans(self, language_code1, language_code2):
        result = check_dictionary(language_code1, language_code2)
        result = list(result)
        result[-1] = "{:2.2f}%%".format(result[-1])
        t = PrettyTable()
        t.field_names = ["Usable Words", "Unusable Words", "Total", "% Usable"]
        t.add_row(result)
        return t

    def print_thresholds(self):
        t = PrettyTable()
        t.field_names = ["Translation", "Result"]
        s = "{} to {}"

        for i, lang1 in enumerate(self.langs):
            for j, lang2 in enumerate(self.langs):
                if i != j:
                    row = [s.format(lang1, lang2)]
                    result = self.get_single_threshold_table(i, j)
                    row.append(result)
                    t.add_row(row)

        print(t)







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


    df = df[df["lang1_median_max_cosine"]!=-1]
    df = df[df["lang2_median_max_cosine"]!=-1]

    high_h_df = df[df["h_score"] > homogeneity_threshold]
    low_h_df = df[df["h_score"] <= homogeneity_threshold]

    for i, data in enumerate([high_h_df, low_h_df, df]):
        temp = data[data["lang1_median_max_cosine"] > cosine_threshold]
        rows[i][0] = len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])

        temp = data[data["lang1_median_max_cosine"] <= cosine_threshold]
        rows[i][2] = len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])

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




def threshold_all(homogeneity_threshold, cosine_threshold, langs, file_name):
    "translation goes rows to columns"
    big_table = PrettyTable()
    big_table.field_names = ["Translation", "Result", "Num Nan"]

    s = "{} to {}"

    for lang1 in langs:
        for lang2 in langs:
            if lang1 != lang2:
                row = [s.format(lang1, lang2)]
                sub_t = threshold(homogeneity_threshold, cosine_threshold, lang1, lang2)
                nans = get_nans(lang1, lang2)
                row.append(sub_t)
                row.append(nans)
        big_table.add_row(row)
    with open(file_name,"w+") as f:
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


    df = df[df["lang1_median_max_cosine"]!=-1]
    df = df[df["lang2_median_max_cosine"]!=-1]

    high_h_df = df[df["h_score"] > homogeneity_threshold]
    low_h_df = df[df["h_score"] <= homogeneity_threshold]

    for i, data in enumerate([high_h_df, low_h_df, df]):
        temp = data[data["lang1_median_max_cosine"] > cosine_threshold]
        rows[i][0] = len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])

        temp = data[data["lang1_median_max_cosine"] <= cosine_threshold]
        rows[i][2] = len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])
        rows[i][1] += len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])

        rows[i][3] = len(data)


    if  len(df)!= 0:
        ratio = rows[2][0]/rows[-1][ -1]
        return ratio
    else:
        return -1


def rank_all(homogeneity_threshold, cosine_threshold, langs):
   for lang in langs:
        worst, best, ratio_list = rank(homogeneity_threshold, cosine_threshold, lang, langs)

        print(lang.upper())
        print("Worst", worst, "Best", best)
        for label, ratio in ratio_list:
            print("{} : {}, ".format(label, ratio), end="")
        print("\n\n")

def rank(homogeneity_threshold, cosine_threshold, lc, langs):
    ratio_list= []
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

if __name__=="__main__":
    main()
