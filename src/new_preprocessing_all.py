import re
import os
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from lxml import objectify

CUSTOM_STOPWORDS = ['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta', 'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je', 'dame', 'herren', 'damen', 'abgeordnete', 'abgeordneter', 'abgeordneten', 'bundestagsabgeordneten', 'bundestagsabgeordnete', 'bundestagsabgeordneter', 'präsident', 'staatssekretär']

SPEECH_START_INDICATORS = ['das wort hat ', 'erteile ich das wort ', 'erteile ich dem ', 'erteile ich der ', 'ich rufe die frage ', 'rufe ich die frage', 'rufe ich auch die frage', 'ich erteile das wort', 'ich erteile ihm das wort', 'ich erteile ihm dazu das wort', 'ich erteile ihnen das wort', 'ich erteile jetzt']

TAGESORDNUNGS_INDICATORS = [' rufe den punkt', ' zusatzpunkt ', ' der tagesordnung ', ' heutigen tagesordnung ', 'zur tagesordnung', ' drucksache ']

FILTER_INDICATORS = ['kohle', 'kernkraft', 'energie', 'treibhaus', 'atom', 'umwelt', 'klimawandel', 'erderwärmung', 'ökologisch', 'abgase', 'abholzung', 'artenschutz', 'erdatmosphäre', 'bergbau', 'bevölkerungswachstum', 'biogas', 'umweltministerium', 'chemikalien', 'verseuchung', 'co2', 'dürre', 'naturkatastrophe', 'flut', 'elektromobilität', 'emission', 'erdgas', 'erneuerbare energien', 'extremwetter', 'feinstaub', 'fischerei', 'fleischkonsum', 'fossile rohstoffe', 'geothermie', 'gletscher', 'hitzewelle', 'klimaschutz', 'klimaskeptiker', 'lichtverschmutzung', 'luftqualität', 'luftverschmutzung', 'mikroplastik', 'nachhaltigkeit', 'nachwachsende rohstoffe', 'naturbewusstsein', 'naturschutz', 'ökobilanz', 'öl', 'ozon', 'permafrost', 'photovoltaik', 'radioaktiv', 'recycling', 'regenwald', 'ressourcenschonung', 'schadstoffe', 'smog', 'solar', 'strom', 'tschernobyl', 'überfischung', 'umweltpolitik', 'umweltverschmutzung', 'klimagipfel', 'versiegelung', 'dürre']

MINIMAL_TAGES_MATCHES = 1
MINIMAL_MATCHES = 3
MIN_WORD_COUNT = 50

PRAESIDIUM_MEMBERS_PER_BUNDESTAG = {
        'bundestag_speeches_pp1.csv': ['Dr. Erich Köhler', 'Dr. Hermann Ehlers', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Hermann Schäfer'],
        'bundestag_speeches_pp2.csv': ['Dr. Hermann Ehlers', 'Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Ludwig Schneider', 'Dr. Max Becker (Hersfeld)'],
        'bundestag_speeches_pp3.csv': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Max Becker (Hersfeld)', 'Dr. Thomas Dehler', 'Dr. Victor-Emanuel Preusker'],
        'bundestag_speeches_pp4.csv': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Erwin Schoettle', 'Dr. Thomas Dehler'],
        'bundestag_speeches_pp5.csv': ['Dr. Eugen Gerstenmaier', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Maria Probst', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Karl Mommer', 'Erwin Schoettle', 'Dr. Thomas Dehler', 'Walter Scheel'],
        'bundestag_speeches_pp6.csv': ['Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        'bundestag_speeches_pp7.csv': ['Dr. Annemarie Renger', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        'bundestag_speeches_pp8.csv': ['Dr. Karl Carstens (Fehmarn)', 'Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Dr. Annemarie Renger', 'Dr. Hermann Schmitt-Vockenhausen', 'Georg Leber', 'Liselotte Funck', 'Richard Wurbs'],
        'bundestag_speeches_pp9.csv': ['Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Heinrich Windelen', 'Dr. Annemarie Renger', 'Georg Leber', 'Richard Wurbs'],
        'bundestag_speeches_pp10.csv': ['Dr. Rainer Barzel', 'Dr. Philipp Jenninger', 'Richard Stücklen', 'Dr. Annemarie Renger', 'Heinz Westphal', 'Richard Wurbs', 'Dieter-Julius Cronenberg (Arnsberg)'],
        'bundestag_speeches_pp11.csv': ['Dr. Philipp Jenninger', 'Dr. Rita Süssmuth', 'Richard Stücklen', 'Dr. Annemarie Renger', 'Heinz Westphal', 'Dieter-Julius Cronenberg (Arnsberg)'],
        'bundestag_speeches_pp12.csv': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Helmuth Becker (NienBerge)', 'Renate Schmidt (Nürnberg)', 'Dieter-Julius Cronenberg (Arnsberg)'],
        'bundestag_speeches_pp13.csv': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Michaela Geiger', 'Hans-Ulrich Klose', 'Dr. Antje Vollmer', 'Dr. Burkhard Hirsch'],
        'bundestag_speeches_pp14.csv': ['Dr. h.c. Wolfgang Thierse', 'Dr. Rudolf Seiters', 'Anke Fuchs (Köln)', 'Petra Bläss', 'Dr. Antje Vollmer', 'Dr. Hermann Otto Solms'],
        'bundestag_speeches_pp15.csv': ['Dr. h.c. Wolfgang Thierse', 'Dr. Norbert Lammert', 'Dr. h.c. Susanne Kastner', 'Dr. Antje Vollmer', 'Dr. Hermann Otto Solms'],
        'bundestag_speeches_pp16.csv': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Dr. h.c. Susanne Kastner', 'Dr. h.c. Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        'bundestag_speeches_pp17.csv': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Eduard Oswald', 'Dr. h.c. Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        'bundestag_speeches_pp18.csv': ['Dr. Norbert Lammert', 'Peter Hintze', 'Michaela Noll', 'Johannes Singhammer', 'Dr. h.c. Edelgard Bulmahn', 'Ulla Schmidt (Aachen)', 'Petra Pau', 'Claudia Roth (Augsburg)'],
        'bundestag_speeches_pp19.csv': ['Dr. Wolfgang Schäuble', 'Dr. Hans-Peter Friedrich (Hof)', 'Thomas Oppermann', 'Petra Pau', 'Claudia Roth (Augsburg)', 'Wolfgang Kubicki']
    }
lemmatizer = WordNetLemmatizer()
with open('src/MDB_STAMMDATEN.xml', 'rb') as fd:
    root = objectify.fromstring(fd.read())
mdb_names = set()
for mdb in root.iterchildren():
    if mdb.tag != 'MDB':
        continue
    for names in mdb.iterchildren():
        if names.tag != 'NAMEN':
            continue
        for name in names.iterchildren():
            if name.tag != 'NAME':
                continue
            for partname in name.iterchildren():
                if partname.tag == 'VORNAME' or partname.tag == 'NACHNAME':
                    name = str(partname).lower()
                    mdb_names.add(name)
                    if name[0] == 'l':
                        print(name)
print(len(mdb_names))
# def write_to_file(file_name, output):
#     with open('outfile', 'wb') as fp:
#         pickle.dump(output, fp)

# def read_from_file(file_name):
#     with open ('outfile', 'rb') as fp:
#         return pickle.load(fp)


def get_wordnet_pos(word):
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
                .replace('-', ' ').replace('--', ' ').replace('_', ' ').replace('*', '').replace('–', '').replace('„', '').replace('~', '')
            new_col.append(str_el.lower())

    return new_col


def prepare_words(col):
    new_col = []
    for idx, speech in enumerate(col):
        stop_words = set(stopwords.words('german') + CUSTOM_STOPWORDS + list(mdb_names))
        new_col.append([lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) for tagged in speech if lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1])) not in stop_words
                        and not (re.search('\d+', tagged[0])) and len(lemmatizer.lemmatize(tagged[0], get_wordnet_pos(tagged[1]))) > 3])
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
            if tagged[0][1] in ['NN', 'NNS', 'CD']:
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
            tages_key_match = 0
            for key in FILTER_INDICATORS:
                if key in speech['Speech text']:
                    tages_key_match += 1
            if tages_key_match >= MINIMAL_TAGES_MATCHES:
                write = True
        individ_key_match = 0
        for key in FILTER_INDICATORS:
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
        result.append( {
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
        result.append( {
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
