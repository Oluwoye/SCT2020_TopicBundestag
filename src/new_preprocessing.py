import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from commons import get_chair, get_custom_stopwords, get_tagesordnung_indicators, get_filter_indicators, \
    filter_by_pos, replace_special_characters, get_mdb_names, get_wordnet_pos

MINIMAL_TAGES_MATCHES = 2
MINIMAL_MATCHES = 5
MIN_WORD_COUNT = 50

lemmatizer = WordNetLemmatizer()
mdb_names = get_mdb_names()


def prepare_words(col):
    new_col = []
    for idx, speech in enumerate(col):
        stop_words = set(stopwords.words('german') + get_custom_stopwords() + list(mdb_names))
        new_col.append([lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) for tagged in speech if
                        lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) not in stop_words
                        and not (re.search('\d+', tagged[0])) and len(lemmatizer.lemmatize(tagged[0],
                                                                                           get_wordnet_pos(
                                                                                               tagged[1]))) > 3])
        print('lemmatized ', idx + 1, '/', len(col), ' speeches')
    return new_col


def preprocess(col):
    print('init')
    col = filter_by_pos(col)
    print('tagged')
    col = prepare_words(col)
    print('lemmatized')
    col = concatenate_to_document(col)
    print('done preprocessing speech')
    return col


def concatenate_to_document(col):
    new_col = []
    for word_list in col:
        document = ""
        for word in word_list:
            document += ' ' + word
        new_col.append(document)
    return new_col


def merge_speeches(df, filename):
    init_speeches = df.to_dict('records')
    write = False
    speeches = []
    filter_indicators = get_filter_indicators()
    for speech in init_speeches:
        if speech['Speaker'] in get_chair()[filename.split('.')[0].split('_')[1]] and any(
                substring in speech['Speech text'] for substring in get_tagesordnung_indicators()):
            write = False
            tages_key_match = 0
            for key in filter_indicators:
                tages_key_match += speech['Speech text'].count(key)
            if tages_key_match >= MINIMAL_TAGES_MATCHES:
                write = True
        individ_key_match = 0
        for key in filter_indicators:
            individ_key_match += speech['Speech text'].count(key)
        if individ_key_match < MINIMAL_MATCHES:
            continue
        if write:
            speeches.append(speech)
    if len(speeches) == 0:
        print('filtered out :', df.iloc[0]['Date'])
        return []
    result = []
    print('tagesordnungspunkte found: ', len(speeches))
    for speech in speeches:
        if len(speech['Speech text']) <= MIN_WORD_COUNT:
            continue
        if isinstance(speech['Speaker'], str) and 'CDU/CSU' in speech['Speaker']:
            speech['Speaker'] = ''
        result.append({
            'Speech DB ID': speech['Speech DB ID'],
            'Date': speech['Date'],
            'Speaker': speech['Speaker'],
            'Speaker party': speech['Speaker party'],
            'Speech text': speech['Speech text']
        })
    print(len(result), ' RESULTS!')
    print('processing complete for: ', df.iloc[0]['Date'])
    return result


def main():
    path = 'data/input/bundestag_speeches_from_10'
    output_path = 'data/merged/final_single/'
    for filename in os.listdir(path):
        if filename != 'bundestag_19.csv':
            continue
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        dates = bundestag['Date'].unique()
        merged_speeches = []
        bundestag = bundestag.apply(
            lambda col: replace_special_characters(col) if col.name == "Speech text"
            else col)
        for date in dates:
            merged_speeches += merge_speeches(bundestag[bundestag['Date'] == date], filename)
        print('done merging')
        df = pd.DataFrame(merged_speeches)
        df = df.apply(
            lambda col: preprocess(col) if col.name == "Speech text"
            else col)
        print('done preprocessing')
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path + filename)
        df.to_csv(output_file)
        print("Finished processing of: " + filename)


if __name__ == '__main__':
    main()
