from utils import *
from dictionary import get_foreign_to_foreign_dictionary, get_translation_dictionary
from homogeneity import run_k_means
from med_max_cosine import median_max_cosine
from sklearn.cluster import k_means
from sklearn.metrics.cluster import homogeneity_score
import numpy as np

import PATHS

NEGATIVE_HOMOGENEITY_SAMPLE_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Negative-Sample-Scores/{}_to_{}_negative.tsv"
NEGATIVE_MED_MAX_COSINE_SAMPLE_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Negative-Sample-Scores/{}_negative.tsv"

NEGATIVE_ENGLISH_MED_MAX_COSINE_SAMPLE_PATH = PATHS.SCORE_RESULTS_FOLDER + "/Negative-Sample-Scores/Negative-En-Med-Max-Cosine/en_{}_negative.tsv"

def create_negative_homogeneity_sample(language_code1, language_code2, num_samples=50):
    # In case the user passes in 'en' for the language code
    if language_code1 == "en":
        translation_d = get_translation_dictionary(
            language_code2, inverted=True)  # en to foreign
        # en with foreign lang package
        cnn_index1 = get_cnn_index("en", language_code2)
        # foreign with foreign lang package
        cnn_index2 = get_cnn_index(language_code2, language_code2)
    elif language_code2 == "en":
        translation_d = get_translation_dictionary(
            language_code1)  # foreign to en
        # foreign with foreign lang package
        cnn_index1 = get_cnn_index(language_code1, language_code1)
        # en with foreign lang package
        cnn_index2 = get_cnn_index("en", language_code1)
    else:
        translation_d = get_foreign_to_foreign_dictionary(
            language_code1, language_code2)  # foreign to foreign
        # foreign with foreign package
        cnn_index1 = get_cnn_index(language_code1, language_code1)
        # foreign with foreign package
        cnn_index2 = get_cnn_index(language_code2, language_code2)

    keys1 = list(cnn_index1.keys()) # lang1 words in a list
    keys2 = list(cnn_index2.keys()) # lang2 words in a list

    scores = np.zeros((num_samples, 3), dtype=object)

    count = 0
    for i in range(num_samples):
        while True:
            count += 1
            if count > 100:
                count = 0
                return

            key1 = keys1[np.random.randint(0, high=len(keys1))] # word in lang1
            key2 = keys2[np.random.randint(0, high=len(keys2))] # word in lang2
            
            if key1 not in translation_d:
                count += 1
                if count > 400:
                    print(language_code1, language_code2)
                    print(key1, key2)
                    break
                continue

            if translation_d[key1] == key2:
                continue
                
            file_number1 = cnn_index1[key1]
            file_number2 = cnn_index2[key2]

            if language_code1 == "en":
                # english with foreign package
                matrix1 = get_matrix(file_number1, "en", language_code2)
                # foreign from foreign package
                matrix2 = get_matrix(
                    file_number2, language_code2, language_code2)
            elif language_code2 == "en":
                # foreign from foreign package
                matrix1 = get_matrix(
                    file_number1, language_code1, language_code1)
                # english from foreign package
                matrix2 = get_matrix(file_number2, "en", language_code1)
            else:
                matrix1 = get_matrix(
                    file_number1, language_code1, language_code1)
                matrix2 = get_matrix(
                    file_number2, language_code2, language_code2)

            if matrix1 is None or matrix2 is None:
                continue

            m1, n1 = matrix1.shape
            m2, n2 = matrix2.shape

            if n1 != 4096 or n2 != 4096:
                continue

            true_labels = np.concatenate(
                (np.zeros((1, m1)), np.ones((1, m2))), axis=1)
            data = np.concatenate((matrix1, matrix2), axis=0)
            centroid, pred_labels, _ = k_means(data, n_clusters=2)
            h_score = homogeneity_score(
                true_labels.flatten(), pred_labels.flatten())
            
            scores[i, :] = np.array([key1, key2, h_score])

            break

    file_name = NEGATIVE_HOMOGENEITY_SAMPLE_PATH.format(language_code1, language_code2)
    np.savetxt(file_name, scores, delimiter='\t',
               fmt='%s', encoding='utf-8')


def create_negative_cosine_sample(language_code, language_package=None, num_files=4):
    if language_package is None:
        language_package = language_code
        file_name = NEGATIVE_MED_MAX_COSINE_SAMPLE_PATH.format(language_code)
    elif language_code == "en":
        file_name = NEGATIVE_ENGLISH_MED_MAX_COSINE_SAMPLE_PATH.format(
            language_package)
    else:
        raise AssertionError("Can only pass language_package with english language code")

    cnn_index = get_cnn_index(language_code, language_package)
    keys = list(cnn_index.keys())

    # Allocates space for combined matrix
    partition_size = 100//num_files
    combined_matrix = np.zeros((partition_size * num_files, 4096))

    words = [""] * num_files

    for i in range(num_files):
        while True:
            key = keys[np.random.randint(0, high=len(keys))]
            file_number = cnn_index[key]
            matrix = get_matrix(file_number, language_code, language_package)

            # Ensures matrix is usable
            if matrix is None:
                continue

            m, n = matrix.shape
            if n != 4096 or m < 100/num_files:
                continue
            break
        
        words[i] = key

        start = i * partition_size
        end = (i + 1) * partition_size
        combined_matrix[start:end, :] = matrix[:partition_size, :] 
    
    score = median_max_cosine(combined_matrix)[0]

    
    with open(file_name, "w+", encoding="utf-8") as f:
        f.write(str(score) + "\n")
        f.write(str(words) + "\n")
        f.write("Number of files used: {}".format(num_files))


def combine_english_negative(langs, num_files=4):
    running_sum = 0
    words = []
    for lang in langs:
        if lang =="en":
            continue
        file_name = NEGATIVE_ENGLISH_MED_MAX_COSINE_SAMPLE_PATH.format(lang)
        with open(file_name, "r", encoding="utf-8") as f:
            line = f.readline()
            running_sum += float(line.strip("\n"))
            words += f.readline().strip("[]\n").replace(" ", "").replace("'", "").split(",")

    score = running_sum/len(langs)

    file_name = NEGATIVE_MED_MAX_COSINE_SAMPLE_PATH.format("en")
    with open(file_name, "w+", encoding="utf-8") as f:
        f.write(str(score) + "\n")
        f.write(str(words) + "\n")
        f.write(str(len(words)))

def main():
    for lang1 in PATHS.LANGS:
        for lang2 in PATHS.LANGS:
            if lang1 != lang2:
                create_negative_homogeneity_sample(lang1, lang2)
                print(lang1, lang2)

    # for lang in PATHS.LANGS:
    #     if lang == "en":
    #         for lang_package in PATHS.LANGS[1:]:
    #             create_negative_cosine_sample(lang, language_package=lang_package)
    #     else:
    #         create_negative_cosine_sample(lang)

    #     print(lang)


    # combine_english_negative(PATHS.LANGS)
    


if __name__ == "__main__":
    main() 
