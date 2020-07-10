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
NEGATIVE_SAMPLE_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Negative-Sample-Scores/{}"

NEGATIVE_HOMOGENEITY_PATH = NEGATIVE_SAMPLE_PATH.format("/Negative-Homogeneity-10/{}_to_{}_negative.tsv")
NEGATIVE_COSINE_PATH = NEGATIVE_SAMPLE_PATH.format("Negative-Cosine-4/{}_negative.tsv")


class Threshold:
    def __init__(self, languages):
        self.ignore_less_than = 10
        self.langs = languages
        self.num2lang = {i: lang for i, lang in enumerate(languages)}
        self.lang2num = {lang: i for i, lang in enumerate(languages)}
        self.num_langs = len(languages)
        self.thresholds = np.zeros(
            (len(languages), len(languages)), dtype=object)
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

        cosine_threshold1 = self.get_cosine_threshold(lc1)
        cosine_threshold2 = self.get_cosine_threshold(lc2)
        homogeneity_threshold = self.get_homogeneity_threshold(lc1, lc2)

        # cosine_threshold1 = 0.45
        # cosine_threshold2 = 0.45
        # homogeneity_threshold = 0.15


        if cosine_threshold1 is None or cosine_threshold2 is None or homogeneity_threshold is None:
            return np.ones((3, 4), dtype=int) * -3

        cosine_threshold = (cosine_threshold1 + cosine_threshold2)/2

        cosine_d1 = read_cosine_TSV(lc1)
        cosine_d2 = read_cosine_TSV(lc2)

        df = read_homogeneity_TSV(lc1, lc2)

        if cosine_d1 is None or cosine_d2 is None or df is None:
            return np.ones((3, 4), dtype=int) * -2

        

        rows = np.zeros((3, 4), dtype=int).tolist()

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
            rows[i][0] = len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])
            rows[i][1] += len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])

            temp = data[data["lang1_median_max_cosine"] <= cosine_threshold]
            rows[i][2] = len(temp[temp["lang2_median_max_cosine"] <= cosine_threshold])
            rows[i][1] += len(temp[temp["lang2_median_max_cosine"] > cosine_threshold])

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

        print("Hello")
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
        

    def get_homogeneity_threshold(self, language_code1, language_code2):
        path = NEGATIVE_HOMOGENEITY_PATH.format(language_code1, language_code2)

        if not os.path.isfile(path):
            print("could not find", language_code1, language_code2)
            return None

        average_threshold = 0
        num_scores = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                score = float(line.strip("\n").split("\t")[-1])
                average_threshold += score
                num_scores += 1

        return average_threshold/num_scores

    def get_cosine_threshold(self, language_code):
        path = NEGATIVE_COSINE_PATH.format(language_code)

        if not os.path.isfile(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            score = float(f.readline().strip("\n"))

        return score


        


def main():
    obj = Threshold(PATHS.LANGS)
    # obj.print_thresholds()
    obj.print_ranks()

    print(obj.ratios[:, obj.lang2num["fr"]])
    print(obj.ratios[:, obj.lang2num["ro"]])

if __name__ == "__main__":
    main()
