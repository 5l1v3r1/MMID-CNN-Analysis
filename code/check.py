import PATHS
from utils import *
import numpy as np
from dictionary import get_foreign_to_foreign_dictionary, get_translation_dictionary
import matplotlib.pyplot as plt

LANGS = PATHS.LANGS
counts = np.zeros((len(LANGS), len(LANGS)))
totals = np.ones((len(LANGS), len(LANGS)))

lang2num = {lang: i for i, lang in enumerate(LANGS)}
num2lang = dict(enumerate(LANGS))

word_phrases = []
good_words = []

for i, lang1 in enumerate(LANGS):
    for j, lang2 in enumerate(LANGS):
        if i == j:
            continue

        # print(lang1, lang2, end="\t")
        if lang1 == "en":
            d = get_translation_dictionary(lang2, inverted=True) # en to foreign
            cnn_index1 = get_cnn_index("en", lang2) # en with foreign lang package
            cnn_index2 = get_cnn_index(lang2, lang2) # foreign with foreign lang package
        elif lang2 == "en":
            d = get_translation_dictionary(lang1) # foreign to en
            cnn_index1 = get_cnn_index(lang1, lang1) # foreign with foreign lang package
            cnn_index2 = get_cnn_index("en", lang1) # en with foreign lang package
        else:
            d = get_foreign_to_foreign_dictionary(lang1, lang2) # foreignt to foreign
            cnn_index1 = get_cnn_index(lang1, lang1) # foreign with foreign package
            cnn_index2 = get_cnn_index(lang2, lang2) # foreign with foreign package

        count = 0
        totals[i, j] = len(d)
        if totals[i, j] == 0:
            print(num2lang[i], num2lang[j])
            totals[i, j] = -2
        for k, word1 in enumerate(d.keys()):
            word2 = d[word1]

            word1ls = word1.split()
            word2ls = word2.split()
            
            if len(word1ls) > 1:
                word_phrases.append(word1ls)
                word1ls.reverse()
                for word in word1ls:
                    if word in cnn_index1:
                        good_words.append(word)
                        break

                word1 = word
                
            
            if len(word2ls) > 1:
                word_phrases.append(word2ls)
                word2ls.reverse()
                for word in word2ls:
                    if word in cnn_index2:
                        good_words.append(word)
                        break
                
                word2 = word


            if word1 in cnn_index1 and word2 in cnn_index2:
                count +=1
        
        counts[i, j] = count

np.fill_diagonal(counts, -1)

PLOT_PATH = "/project/multilm/nikzad/Analyze-CNN-code/plots/"


def plot_hist(data, title, num_bins=50, num_ticks=11):
    path = PLOT_PATH + str(title) + ".png"

    bins = plt.hist(data, bins=num_bins)
    plt.title(title)
    plt.ylabel("Number of Occurrences")
    plt.xlabel(title.title())
    plt.xticks(np.linspace(0, max(data), num_ticks))
    plt.savefig(path)
    plt.close()
    return bins

plot_hist(counts.flatten(), "raw_counts_reverse")

percent = counts/totals * 100
np.fill_diagonal(percent,300)

# print(np.argmin(percent, axis=0))
# print(np.argmin(percent, axis=1))
plot_hist(percent.flatten(), "percent_counts_reverse")

print(len(word_phrases))
# for i in range(len(word_phrases)):
#     print(word_phrases[i], good_words[i])
