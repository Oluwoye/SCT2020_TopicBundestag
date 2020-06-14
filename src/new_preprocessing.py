import re
import os
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet



CUSTOM_STOPWORDS = ['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta', 'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je', 'dame', 'herren', 'damen', 'abgeordnete', 'abgeordneter', 'abgeordneten']

SPEECH_START_INDICATORS = ['das wort hat ', 'erteile ich das wort ', 'erteile ich dem ', 'erteile ich der ', 'ich rufe die frage ', 'rufe ich die frage', 'rufe ich auch die frage', 'ich erteile das wort', 'ich erteile ihm das wort', 'ich erteile ihm dazu das wort', 'ich erteile ihnen das wort', 'ich erteile jetzt']

TAGESORDNUNGS_INDICATORS = [' rufe den punkt', ' zusatzpunkt ', ' der tagesordnung ', ' heutigen tagesordnung ', 'zur tagesordnung', ' drucksache ']

FILTER_INDICATORS = ['kohle', 'kernkraft', 'energie', 'treibhaus', 'atom', 'umwelt', 'klimawandel', 'erderwärmung', 'ökologisch', 'abgase', 'abholzung', 'artenschutz', 'erdatmosphäre', 'bergbau', 'bevölkerungswachstum', 'biogas', 'umweltministerium', 'chemikalien', 'verseuchung', 'co2', 'dürre', 'naturkatastrophe', 'flut', 'elektromobilität', 'emission', 'erdgas', 'erneuerbare energien', 'extremwetter', 'feinstaub', 'fischerei', 'fleischkonsum', 'fossile rohstoffe', 'geothermie', 'gletscher', 'hitzewelle', 'klimaschutz', 'klimaskeptiker', 'lichtverschmutzung', 'luftqualität', 'luftverschmutzung', 'mikroplastik', 'müll', 'nachhaltigkeit', 'nachwachsende rohstoffe', 'naturbewusstsein', 'naturschutz', 'ökobilanz', 'öl', 'ozon', 'permafrost', 'photovoltaik', 'radioaktiv', 'recycling', 'regenwald', 'ressourcenschonung', 'schadstoffe', 'smog', 'solar', 'strom', 'tschernobyl', 'überfischung', 'umweltpolitik', 'umweltverschmutzung', 'klimagipfel', 'versiegelung', 'abfall']

MINIMAL_MATCHES = 2

PRAESIDIUM_MEMBERS_PER_BUNDESTAG = {
        'bundestag_speeches_pp1.csv': ['Erich Köhler', 'Hermann Ehlers', 'Carlo Schmid', 'Hermann Schäfer'],
        'bundestag_speeches_pp2.csv': ['Hermann Ehlers', 'Eugen Gerstenmaier', 'Richard Jaeger', 'Carlo Schmid', 'Ludwig Schneider', 'Max Becker'],
        'bundestag_speeches_pp3.csv': ['Eugen Gerstenmaier', 'Richard Jaeger', 'Carlo Schmid', 'Max Becker', 'Thomas Dehler', 'Victor-Emanuel Preusker'],
        'bundestag_speeches_pp4.csv': ['Eugen Gerstenmaier', 'Richard Jaeger', 'Carlo Schmid', 'Erwin Schoettle', 'Thomas Dehler'],
        'bundestag_speeches_pp5.csv': ['Eugen Gerstenmaier', 'Kai-Uwe von Hassel', 'Richard Jaeger', 'Maria Probst', 'Carlo Schmid', 'Karl Mommer', 'Erwin Schoettle', 'Thomas Dehler', 'Walter Scheel'],
        'bundestag_speeches_pp6.csv': ['Kai-Uwe von Hassel', 'Richard Jaeger', 'Carlo Schmid', 'Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        'bundestag_speeches_pp7.csv': ['Annemarie Renger', 'Kai-Uwe von Hassel', 'Richard Jaeger', 'Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        'bundestag_speeches_pp8.csv': ['Karl Carstens', 'Richard Stücklen', 'Richard von Weizsäcker', 'Annemarie Renger', 'Hermann Schmitt-Vockenhausen', 'Georg Leber', 'Liselotte Funck', 'Richard Wurbs'],
        'bundestag_speeches_pp9.csv': ['Richard Stücklen', 'Richard von Weizsäcker', 'Heinrich Windelen', 'Annemarie Renger', 'Georg Leber', 'Richard Wurbs'],
        'bundestag_speeches_pp10.csv': ['Rainer Barzel', 'Philipp Jenninger', 'Richard Stücklen', 'Annemarie Renger', 'Heinz Westphal', 'Richard Wurbs', 'Dieter-Julius Cronenberg'],
        'bundestag_speeches_pp11.csv': ['Philipp Jenninger', 'Rita Süssmuth', 'Richard Stücklen', 'Annemarie Renger', 'Heinz Westphal', 'Dieter-Julius Cronenberg'],
        'bundestag_speeches_pp12.csv': ['Rita Süssmuth', 'Hans Klein', 'Helmuth Becker', 'Renate Schmidt', 'Dieter-Julius Cronenberg'],
        'bundestag_speeches_pp13.csv': ['Rita Süssmuth', 'Hans Klein', 'Michaela Geiger', 'Hans-Ulrich Klose', 'Antje Vollmer', 'Burkhard Hirsch'],
        'bundestag_speeches_pp14.csv': ['Wolfgang Thierse', 'Rudolf Seiters', 'Anke Fuchs', 'Petra Bläss', 'Antje Vollmer', 'Hermann Otto Solms'],
        'bundestag_speeches_pp15.csv': ['Wolfgang Thierse', 'Norbert Lammert', 'Susanne Kastner', 'Antje Vollmer', 'Hermann Otto Solms'],
        'bundestag_speeches_pp16.csv': ['Norbert Lammert', 'Gerda Hasselfeldt', 'Susanne Kastner', 'Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Hermann Otto Solms'],
        'bundestag_speeches_pp17.csv': ['Norbert Lammert', 'Gerda Hasselfeldt', 'Eduard Oswald', 'Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Hermann Otto Solms'],
        'bundestag_speeches_pp18.csv': ['Norbert Lammert', 'Peter Hintze', 'Michaela Noll', 'Johannes Singhammer', 'Edelgard Bulmahn', 'Ulla Schmidt', 'Petra Pau', 'Claudia Roth'],
        'bundestag_speeches_pp19.csv': ['Wolfgang Schäuble', 'Hans-Peter Friedrich', 'Thomas Oppermann', 'Petra Pau', 'Claudia Roth', 'Wolfgang Kubicki']
    }
lemmatizer = WordNetLemmatizer()
# def write_to_file(file_name, output):
#     with open('outfile', 'wb') as fp:
#         pickle.dump(output, fp)

# def read_from_file(file_name):
#     with open ('outfile', 'rb') as fp:
#         return pickle.load(fp)

def  get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = word[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

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
    for idx, speech in enumerate(col):
        stop_words = set(stopwords.words('german') + CUSTOM_STOPWORDS)
        new_col.append([lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) for tagged in speech if lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) not in stop_words
                        and not (re.search('\d+', tagged[0])) and len(lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1]))) > 1])
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

def filter_by_pos(col):
    new_col = []
    for idx, speech in enumerate(col):
        tokens = word_tokenize(speech)
        new_speech = []
        for word in tokens:
            tagged = nltk.pos_tag([word])
            if tagged[0][1] in ['NN', 'NNS']:
                new_speech.append(tagged[0])
        new_col.append(new_speech)
        print('filtered pos : ', idx + 1, '/', len(col))
    return new_col

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
    for speech in init_speeches:
        if speech['Speaker'] in PRAESIDIUM_MEMBERS_PER_BUNDESTAG[filename] and any(substring in speech['Speech text'] for substring in TAGESORDNUNGS_INDICATORS):
            write = False
            #if any(substring in speech['Speech text'] for substring in FILTER_INDICATORS):
            key_match = 0
            for key in FILTER_INDICATORS:
                if key in speech['Speech text']:
                    key_match += 1
            if key_match >= MINIMAL_MATCHES:    
                write = True
        if write:
            speeches.append(speech)
    if len(speeches) == 0:
        print('filtered out :', df.iloc[0]['Date'])
        return []
    result = []
    print('init length: ', len(speeches))
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
            if len(speech['Speech text']) <= 50:
                continue
            if speech['Speaker'] in selected_result:
                selected_result[speech['Speaker']]['Speech text'] += speech['Speech text']
            else:
                selected_result[speech['Speaker']] = {
                    'Speech DB ID': speech['Speech DB ID'],
                    'Date': speech['Date'],
                    'Speaker': speech['Speaker'],
                    'Speaker party': speech['Speaker party'],
                    'Speech text': speech['Speech text']
                }
        for speaker in selected_result:
            result.append(selected_result[speaker])
    print(len(result), ' RESULTS!')
    print('processing complete for: ', df.iloc[0]['Date'])
    return result

def main():
    path = 'data/input/bundestag_speeches_pp09-14'
    output_path = 'data/merged/'
    for filename in os.listdir(path):
        if filename != 'bundestag_speeches_pp9.csv':
            continue
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        dates = bundestag['Date'].unique()
        merged_speeches = []
        bundestag = bundestag.apply(
            lambda col: replace_special_characters(col) if col.name == "Speech text"
            else col)
        for date in dates:
            merged_speeches += merge_speeches(bundestag[bundestag['Date']==date], filename)
        print('done merging')
        df = pd.DataFrame(merged_speeches)
        df = df.apply(
            lambda col: preprocess(col) if col.name == "Speech text"
            else col)
        print('done preprocessing')
        output_file = os.path.join(output_path + '01'  + filename)
        df.to_csv(output_file)
        print("Finished processing of: " + filename)


if __name__ == '__main__':
    main()
