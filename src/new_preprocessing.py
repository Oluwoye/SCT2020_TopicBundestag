import re
import os
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



CUSTOM_STOPWORDS = ['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta', 'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je']

SPEECH_START_INDICATORS = ['das wort hat ', 'erteile ich das wort ', 'erteile ich dem ', 'erteile ich der ', 'ich rufe die frage ', 'rufe ich die frage', 'rufe ich auch die frage', 'ich erteile das wort', 'ich erteile ihm das wort', 'ich erteile ihm dazu das wort', 'ich erteile ihnen das wort', 'ich erteile jetzt']

# def write_to_file(file_name, output):
#     with open('outfile', 'wb') as fp:
#         pickle.dump(output, fp)

# def read_from_file(file_name):
#     with open ('outfile', 'rb') as fp:
#         return pickle.load(fp)

def replace_special_characters(col):
    new_col = []
    for _, str_el in enumerate(col):
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


def prepare_words(col):
    new_col = []
    lemmatizer = WordNetLemmatizer()
    for _, str_el in enumerate(col):
        word_set = set(word_tokenize(str_el))
        stop_words = set(stopwords.words('german') + CUSTOM_STOPWORDS)
        new_col.append([lemmatizer.lemmatize(word) for word in word_set if lemmatizer.lemmatize(word) not in stop_words
                        and not (re.search('\d+', word)) and len(lemmatizer.lemmatize(word)) > 1])
    return new_col
        
def preprocess(col):
    col = prepare_words(col)
    col = filter_by_pos(col)
    print('done preprocessing speech')
    return concatenate_to_document(col)

def filter_by_pos(col):
    new_col = []
    for speech in col:
        new_col.append([word for word in speech if nltk.pos_tag([word])[0][1] in ['NN', 'NNS']])
    return new_col

def concatenate_to_document(col):
    new_col = []
    for word_list in col:
        document = ""
        for word in word_list:
            document += ' ' + word
        new_col.append(document)
    return new_col

def merge_speeches(df):
    speeches = df.to_dict('records')
    result = []

    start_idx_list = []
    for idx, speech in enumerate(speeches):
        if any(substring in speech['Speech text'] for substring in SPEECH_START_INDICATORS):
            if idx not in start_idx_list:
                start_idx_list.append(idx)
    for i, el in enumerate(start_idx_list):
        if i == len(start_idx_list)-1:
            selected_speeches = speeches[el:len(speeches)]
        else:
            selected_speeches = speeches[el:start_idx_list[i+1]]
        selected_result = {}
        for speech in selected_speeches:
            if speech['Speaker'] in selected_result:
                selected_result[speech['Speaker']]['Speech text'] += speech['Speech text']
            else:
                selected_result[speech['Speaker']] = {
                    'Date': speech['Date'],
                    'Speaker': speech['Speaker'],
                    'Speaker party': speech['Speaker party'],
                    'Speech text': speech['Speech text']
                }
        for speaker in selected_result:
            result.append(selected_result[speaker])
    return result

def main():
    path = 'data/input/bundestag_speeches_pp09-14'
    output_path = 'data/merged/'
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        dates = bundestag['Date'].unique()
        merged_speeches = []
        bundestag = bundestag.apply(
            lambda col: replace_special_characters(col) if col.name == "Speech text"
            else col)
        for date in dates:
            print(date)
            merged_speeches += merge_speeches(bundestag[bundestag['Date']==date])
        print('done merging')
        df = pd.DataFrame(merged_speeches)
        print(df.head())
        df = df.apply(
            lambda col: preprocess(col) if col.name == "Speech text"
            else col)
        print('done preprocessing')
        output_file = os.path.join(output_path + filename)
        bundestag.to_csv(output_file)
        print("Finished processing of: " + filename)


if __name__ == '__main__':
    main()
