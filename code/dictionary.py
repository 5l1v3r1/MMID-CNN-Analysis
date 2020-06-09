# from google.cloud import translate_v2 as translate
from nltk.stem.snowball import SnowballStemmer
from utils import get_cnn_index, get_language_code_mapping
import os
import csv
import PATHS

# path to the spanish to english word translation
RAW_DICTIONARY_PATH = PATHS.RAW_DICTIONARY_PATH + "/dict."
PROCESSED_DICTIONARY_PATH = PATHS.DICTIONARY_PATH + "/processed_dict."
FOREIGN_TO_FOREIGN_PATH = PATHS.DICTIONARY_PATH + "/{}_to_{}.tsv"

def process_dictionary(language_code, overwrite=False):
    """ Creates new dictionary tsv in PROCESSED_DICTIONARY_PATH with the name
    processed_dict.[country code]. The new dictionary will be one to one instead
    of one to many with the best translation according to a stemmed google translate
    translation. Stemmed with NLTK.
    :param dic_path: Str with path to the dictionary path of the foreign language
    """
    dic_path = RAW_DICTIONARY_PATH + language_code

    output_path = PROCESSED_DICTIONARY_PATH + language_code

    if os.path.isfile(output_path):
        num_words_raw = sum(1 for line in open(dic_path, encoding='utf-8'))
        num_words_processed = sum(1 for line in open(output_path, encoding='utf-8'))
        if num_words_raw == num_words_processed and not overwrite:
            print("File already exists and will not be overwritten")
            return
        print("File found and will be overwritten")

    translate_client = translate.Client()

    stemmer = SnowballStemmer("english")

    count = 0

    with open(output_path, 'w+', encoding='utf-8') as o:
        w = csv.writer(o, delimiter='\t')

        with open(dic_path, 'r', encoding='utf-8') as f:
            for line in f:

                # count += 1
                # if count % 100 == 0:
                #     print("COMPLETED {} WORDS SO FAR".format(count))

                words = line.rstrip('\n').split('\t')

                foreign_word = words[0]

                english_word = words[1]

                google_translation = translate_client.translate(foreign_word,
                    source_language=language_code, target_language='en')['translatedText'].lower()

                google_stem = stemmer.stem(google_translation)


                for word in words[1:]:
                    word_stem = stemmer.stem(word)
                    if word == google_translation or word_stem == google_stem:
                        english_word = word
                        break

                w.writerow([foreign_word, english_word, google_translation])

def get_translation_dictionary(language_code, overwrite=False):
    """ Gets dictionary for foreign to english translation
    """
    process_dictionary(language_code, overwrite=overwrite)

    with open(PROCESSED_DICTIONARY_PATH + language_code, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]:(row[1], row[2]) for row in r}

    return d

def create_foreign_to_foreign_dictionary(language_code1, language_code2, overwrite=False):
    dic_path = RAW_DICTIONARY_PATH + language_code1

    output_path = FOREIGN_TO_FOREIGN_PATH.format(language_code1, language_code2)

    if os.path.isfile(output_path):
        num_words_raw = sum(1 for line in open(dic_path, encoding='utf-8'))
        num_words_processed = sum(1 for line in open(output_path, encoding='utf-8'))
        if num_words_raw == num_words_processed and not overwrite:
            print("File already exists and will not be overwritten")
            return
        print("File found and will be overwritten")

    print("Writing to {}".format(output_path))

    translate_client = translate.Client()

    # count = 0

    with open(output_path, 'w+', encoding='utf-8') as o:
        w = csv.writer(o, delimiter='\t')

        with open(dic_path, 'r', encoding='utf-8') as f:
            for line in f:

                # count += 1
                # if count % 100 == 0:
                #     print("COMPLETED {} WORDS SO FAR".format(count))

                words = line.rstrip('\n').split('\t')

                foreign_word = words[0]

                english_word = words[1]

                google_translation = translate_client.translate(foreign_word,
                    source_language=language_code1, target_language=language_code2)['translatedText'].lower()

                w.writerow([foreign_word, google_translation, english_word])

def get_foreign_to_foreign_dictionary(language_code1, language_code2):
    """ Gets dictionary for k_means foreign to foreign translation, give two
    language codes. If one doesn't exist it will be created
    """
    create_foreign_to_foreign_dictionary(language_code1, language_code2)

    filename = FOREIGN_TO_FOREIGN_PATH.format(language_code1, language_code2)
    with open(filename, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]:(row[1], row[2]) for row in r}

    return d


def read_raw_dictionary(language_code):
    """ Reads raw dictionary from language code, used for median max cosine calc """
    with open(RAW_DICTIONARY_PATH + language_code, 'r', encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        d = {row[0]:row[1] for row in r}

    return d

def check_dictionary(language_code1, language_code2):
    """ Checks the dictionary to see how many words can be used for calculations.
    Returns the following:
    (total words, usable, unusable, usable/total_words)
    """
    if language_code1 == language_code2:
        print("Dictionary between same languages does not exist")
        return

    langcode2name = get_language_code_mapping()

    if language_code2 == 'en':
        package_name = langcode2name[language_code1]
    else:
        package_name = langcode2name[language_code2]

    if language_code1 == 'en':
        translation_d = get_translation_dictionary(language_code2)
    elif language_code2 == 'en':
        translation_d = get_translation_dictionary(language_code1)
    else:
        translation_d = get_foreign_to_foreign_dictionary(language_code1, language_code2)

    index = get_cnn_index(language_code2, package_name)
    total = len(translation_d)
    num_usable = 0

    for word, english in translation_d.values():
        if word in index:
            num_usable += 1

    return  num_usable, total - num_usable, total, num_usable/total



def main():
    """ Used for preprocessing dictionaries with GOOGLE API. Cannot parallelize,
    or else will run into API rate_limit errors.
    """
    LANGS = ("en", "fr", "ar", "az", "es", "id", "de", "tr", "hi", "it", "vi", "th", "cy")
    for lc1 in ["cy"]:
        for lc2 in LANGS[8:]:
            print("STARTED {} to {}".format(lc1.upper(), lc2.upper()))
            if lc1 != lc2:
                if lc1 == "en":
                    process_dictionary(lc2)
                elif lc2 == "en":
                    process_dictionary(lc1)
                else:
                    create_foreign_to_foreign_dictionary(lc1, lc2)
            print("FINISHED {} to {}".format(lc1.upper(), lc2.upper()))


if __name__ == '__main__':
    main()
