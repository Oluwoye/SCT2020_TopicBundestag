import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.commons import get_chair, get_custom_stopwords, get_filter_indicators, get_tagesordnung_indicators, \
    replace_special_characters, get_frame_and_dates, filter_by_pos, get_mdb_names, get_wordnet_pos

MINIMAL_TAGES_MATCHES = 1
MINIMAL_MATCHES = 3
MIN_WORD_COUNT = 50

lemmatizer = WordNetLemmatizer()

mdb_names = get_mdb_names()


def prepare_words(col):
    new_col = []
    for idx, speech in enumerate(col):
        stop_words = set(stopwords.words('german') + get_custom_stopwords() + list(mdb_names))
        new_col.append([lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) for tagged in speech if
                        lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) not in stop_words
                        and not (re.search('\d+', tagged[0])) and len(
                            lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1]))) > 3])
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
        if speech['Speaker'] in get_chair()[filename] and \
                any(substring in speech['Speech text'] for substring in get_tagesordnung_indicators()):
            write = False
            tages_key_match = 0
            for key in filter_indicators:
                if key in speech['Speech text']:
                    tages_key_match += 1
            if tages_key_match >= MINIMAL_TAGES_MATCHES:
                write = True
        individ_key_match = 0
        for key in filter_indicators:
            if key in speech['Speech text']:
                individ_key_match += 1
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


def filter_columns(df):
    init_speeches = df.to_dict('records')
    result = []
    for speech in init_speeches:
        result.append({
            'Speech DB ID': speech['Speech DB ID'],
            'Date': speech['Date'],
            'Speaker': speech['Speaker'],
            'Speaker party': speech['Speaker party'],
            'Speech text': speech['Speech text']
        })
    return result


def main():
    path = 'data/input/bundestag_speeches_from_10'
    output_path = 'data/merged/final/'
    speech_contribution = {}
    final_speech_contribution = {}
    max_speeches = 0
    for filename in os.listdir(path):
        if filename != 'bundestag_speeches_pp17.csv':
            continue
        bundestag, dates = get_frame_and_dates(filename, path)
        merged_speeches = []
        for date in dates:
            merged_speeches += merge_speeches(bundestag[bundestag['Date'] == date], filename)
        print('done merging')
        df = pd.DataFrame(merged_speeches)
        speech_contribution[filename] = df
        speech_count = len(speech_contribution[filename].index)
        print('SIZE: ', speech_count)
        if speech_count > max_speeches:
            max_speeches = speech_count
    print('MAXIMUM OF SPEECHES FOUND: ', max_speeches)
    for filename in os.listdir(path):
        if filename != 'bundestag_speeches_pp17.csv':
            continue
        needed_speeches = max_speeches - len(speech_contribution[filename].index)
        if needed_speeches == 0:
            final_speech_contribution[filename] = speech_contribution[filename].apply(
                lambda col: preprocess(col) if col.name == "Speech text"
                else col)
            continue
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        bundestag = bundestag.apply(
            lambda col: replace_special_characters(col) if col.name == "Speech text"
            else col)
        filtered = filter_columns(bundestag)
        df = pd.DataFrame(filtered)
        df_elements = df.sample(n=needed_speeches)
        print('SIZE: ', len(df_elements.index))
        final_speech_contribution[filename] = pd.concat([speech_contribution[filename], df_elements], ignore_index=True)
        final_speech_contribution[filename] = final_speech_contribution[filename].apply(
            lambda col: preprocess(col) if col.name == "Speech text"
            else col)
        print('done preprocessing')

    for contribution in final_speech_contribution:
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path + contribution)
        final_speech_contribution[contribution].to_csv(output_file)
        print("Finished processing of: " + contribution)


if __name__ == '__main__':
    main()
