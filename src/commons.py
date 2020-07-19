import os

import nltk
import pandas as pd
from lxml import objectify
from nltk import word_tokenize
from nltk.corpus import wordnet


def get_chair():
    return {
        '1': ['Dr. Erich Köhler', 'Dr. Hermann Ehlers', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Hermann Schäfer'],
        '2': ['Dr. Hermann Ehlers', 'Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)',
              'Dr. Ludwig Schneider', 'Dr. Max Becker (Hersfeld)'],
        '3': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)',
              'Dr. Max Becker (Hersfeld)', 'Dr. Thomas Dehler', 'Dr. Victor-Emanuel Preusker'],
        '4': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Erwin Schoettle',
              'Dr. Thomas Dehler'],
        '5': ['Dr. Eugen Gerstenmaier', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Maria Probst',
              'Dr. Carlo Schmid (Frankfurt)', 'Dr. Karl Mommer', 'Erwin Schoettle', 'Dr. Thomas Dehler',
              'Walter Scheel'],
        '6': ['Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)',
              'Dr. Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        '7': ['Dr. Annemarie Renger', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Hermann Schmitt-Vockenhausen',
              'Liselotte Funcke'],
        '8': ['Dr. Karl Carstens (Fehmarn)', 'Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Dr. Annemarie Renger',
              'Dr. Hermann Schmitt-Vockenhausen', 'Georg Leber', 'Liselotte Funck', 'Richard Wurbs'],
        '9': ['Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Heinrich Windelen', 'Dr. Annemarie Renger',
              'Georg Leber', 'Richard Wurbs'],
        '10': ['Dr. Rainer Barzel', 'Dr. Philipp Jenninger', 'Richard Stücklen', 'Dr. Annemarie Renger',
               'Heinz Westphal', 'Richard Wurbs', 'Dieter-Julius Cronenberg (Arnsberg)'],
        '11': ['Dr. Philipp Jenninger', 'Dr. Rita Süssmuth', 'Richard Stücklen', 'Dr. Annemarie Renger',
               'Heinz Westphal', 'Dieter-Julius Cronenberg (Arnsberg)'],
        '12': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Helmuth Becker (NienBerge)', 'Renate Schmidt (Nürnberg)',
               'Dieter-Julius Cronenberg (Arnsberg)'],
        '13': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Michaela Geiger', 'Hans-Ulrich Klose', 'Dr. Antje Vollmer',
               'Dr. Burkhard Hirsch'],
        '14': ['Dr. h.c. Wolfgang Thierse', 'Dr. Rudolf Seiters', 'Anke Fuchs (Köln)', 'Petra Bläss',
               'Dr. Antje Vollmer', 'Dr. Hermann Otto Solms'],
        '15': ['Dr. h.c. Wolfgang Thierse', 'Dr. Norbert Lammert', 'Dr. h.c. Susanne Kastner', 'Dr. Antje Vollmer',
               'Dr. Hermann Otto Solms'],
        '16': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Dr. h.c. Susanne Kastner', 'Dr. h.c. Wolfgang Thierse',
               'Petra Pau', 'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        '17': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Eduard Oswald', 'Dr. h.c. Wolfgang Thierse', 'Petra Pau',
               'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        '18': ['Dr. Norbert Lammert', 'Peter Hintze', 'Michaela Noll', 'Johannes Singhammer',
               'Dr. h.c. Edelgard Bulmahn', 'Ulla Schmidt (Aachen)', 'Petra Pau', 'Claudia Roth (Augsburg)'],
        '19': ['Dr. Wolfgang Schäuble', 'Dr. Hans-Peter Friedrich (Hof)', 'Thomas Oppermann', 'Petra Pau',
               'Claudia Roth (Augsburg)', 'Wolfgang Kubicki']
    }


def get_custom_stopwords():
    return ['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta',
            'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je', 'dame', 'herren', 'damen', 'abgeordnete',
            'abgeordneter', 'abgeordneten', 'bundestagsabgeordneten', 'bundestagsabgeordnete',
            'bundestagsabgeordneter', 'präsident', 'staatssekretär']


def get_speech_start_indicators():
    return ['das wort hat ', 'erteile ich das wort ', 'erteile ich dem ', 'erteile ich der ',
            'ich rufe die frage ', 'rufe ich die frage', 'rufe ich auch die frage',
            'ich erteile das wort', 'ich erteile ihm das wort', 'ich erteile ihm dazu das wort',
            'ich erteile ihnen das wort', 'ich erteile jetzt']


def get_tagesordnung_indicators():
    return [' rufe den punkt', ' zusatzpunkt ', ' der tagesordnung ', ' heutigen tagesordnung ',
            'zur tagesordnung', ' drucksache ']


def get_filter_indicators():
    return ['kohle', 'kernkraft', 'energie', 'treibhaus', 'atom', 'umwelt', 'klimawandel', 'erderwärmung',
            'ökologisch', 'abgase', 'abholzung', 'artenschutz', 'erdatmosphäre', 'bergbau',
            'bevölkerungswachstum', 'biogas', 'umweltministerium', 'chemikalien', 'verseuchung', 'co2',
            'dürre', 'naturkatastrophe', 'flut', 'elektromobilität', 'emission', 'erdgas',
            'erneuerbare energien', 'extremwetter', 'feinstaub', 'fischerei', 'fleischkonsum',
            'fossile rohstoffe', 'geothermie', 'gletscher', 'hitzewelle', 'klimaschutz', 'klimaskeptiker',
            'lichtverschmutzung', 'luftqualität', 'luftverschmutzung', 'mikroplastik', 'nachhaltigkeit',
            'nachwachsende rohstoffe', 'naturbewusstsein', 'naturschutz', 'ökobilanz', 'öl', 'ozon',
            'permafrost', 'photovoltaik', 'radioaktiv', 'recycling', 'regenwald', 'ressourcenschonung',
            'schadstoffe', 'smog', 'solar', 'strom', 'tschernobyl', 'überfischung', 'umweltpolitik',
            'umweltverschmutzung', 'klimagipfel', 'versiegelung', 'dürre']


def replace_special_characters(col):
    new_col = []
    for _, str_el in enumerate(col):
        if str_el is None or (not isinstance(str_el, str)):
            new_col.append('')
        else:
            str_el = str_el.replace('.', '').replace(';', '').replace(',', '').replace('?', '').replace('!', '') \
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('{', '') \
                .replace('}', '').replace('&', '').replace('%', '').replace('/', '').replace('\\', '') \
                .replace('\'', ' ').replace('´', ' ').replace('`', ' ').replace(':', '').replace('"', '') \
                .replace('-', ' ').replace('--', ' ').replace('_', ' ').replace('*', '').replace('–', '').replace('„',
                                                                                                                  '').replace(
                '~', '')
            new_col.append(str_el.lower())

    return new_col


def get_frame_and_dates(filename, path):
    file = os.path.join(path, filename)
    bundestag = pd.read_csv(file)
    dates = bundestag['Date'].unique()
    bundestag = bundestag.apply(
        lambda col: replace_special_characters(col) if col.name == "Speech text"
        else col)
    return bundestag, dates


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


def get_mdb_names():
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


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = word[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
