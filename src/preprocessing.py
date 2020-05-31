import re
import os
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



CUSTOM_STOPWORDS = ['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta', 'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je']

def replace_special_characters(col):
    new_col = []
    for i, str_el in enumerate(col):
        if str_el is None or (not isinstance(str_el, str)):
            new_col.append('')
        else:
            str_el = str_el.replace('.', '').replace(';', '').replace(',', '').replace('?', '').replace('!', '')\
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('{', '')\
                .replace('}', '').replace('&', '').replace('%', '').replace('/', '').replace('\\', '')\
                .replace('\'', ' ').replace('´', ' ').replace('`', ' ').replace(':', '').replace('"', '')\
                .replace('-', ' ').replace('--', ' ').replace('_', ' ').replace('*', '').replace('–', '').replace('„', '')
            new_col.append(str_el.lower())

    return new_col


def replace_umlauts(col):
    new_col = []
    for i, str_el in enumerate(col):
        str_el = str_el.replace('ß', 's').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'üe')\
            .replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
        new_col.append(str_el)
    return new_col


def prepare_words(col):
    new_col = []
    lemmatizer = WordNetLemmatizer()
    for i, str_el in enumerate(col):
        word_set = set(word_tokenize(str_el))
        stop_words = set(stopwords.words('german') + CUSTOM_STOPWORDS)
        new_col.append([lemmatizer.lemmatize(word) for word in word_set if lemmatizer.lemmatize(word) not in stop_words
                        and not (re.search('\d+', word)) and len(lemmatizer.lemmatize(word)) > 1])
    return new_col


def concatenate_to_document(col):
    new_col = []
    for word_list in col:
        document = ""
        for word in word_list:
            document += ' ' + word
        new_col.append(document)
    return new_col

# def filter_by_keyword_window(col, filter_keywords, window_radius):
    # keywords_by_idx = [i for i, word in col if word in filter_keywords]
    # neighbors = []
#     for idx in keywords_by_idx:
        # neighbors.append(col[idx-window_radius:idx+window_radius+1])
        
def preprocess_col(col):
    col = replace_special_characters(col)
    col = prepare_words(col)
    #naive filter by only allowing text from window around keywords -> duplicates words when keywords overlap
    # window_radius = 30
    # filter_keywords = ['kohle', 'kernkraft', 'kernenergie', 'öl', 'umweltpolitik']
    # col = filter_by_keyword_window(col, filter_keywords, window_radius)
    return concatenate_to_document(col)
    # col = concatenate_to_document(col)
    # return replace_umlauts(col)


def main():
    path = 'data/input/bundestag_speeches_pp09-14'
    output_path = 'data/preprocessed/'
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        bundestag = bundestag.apply(
            lambda col: preprocess_col(col) if col.name == "Speech text" #or col.name == "Interjection content"
            else col)
        output_file = os.path.join(output_path + filename)
        bundestag.to_csv(output_file)
        print("Finished processing of: " + filename)


if __name__ == '__main__':
    main()
