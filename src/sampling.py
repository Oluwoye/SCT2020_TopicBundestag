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

CUSTOM_STOPWORDS = list(set(['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta',
            'kollegin', 'herrn', 'ab', 'wort', 'wyhl', 'je', 'dame', 'herren', 'damen', 'abgeordnete',
            'abgeordneter', 'abgeordneten', 'bundestagsabgeordneten', 'bundestagsabgeordnete',
            'bundestagsabgeordneter', 'präsident', 'staatssekretär', 'müssen', 'mehr', 'schon', 'heute', 'sagen',
            'diesis', 'geht', 'jahren', 'gibt', 'dafür', 'rufe', 'eröffne', 'bereits', 'neuen', 'allerdings',
            'wurden', 'etwa', 'zusammenhang', 'gegenüber', 'folgen', 'daher', 'besteht', 'wichtige', 'sowie',
            'kommen', 'einzelnen', 'übrigen', 'läßt', 'falsch', 'seien', 'versuchen', 'bereit', 'sollen', 'gute',
            'braucht', 'tagen', 'woche', 'innerhalb', 'erkennen', 'hoffe', 'verstehen', 'wenigen', 'notwendige',
            'zuletzt', 'großen', 'nächsten', 'setzen', 'hoch', 'spürbar', 'kolleginnen', 'kollegen', 'worden',
            'zwei', 'denen', 'nehmen', 'herumkommen', 'inmitten', 'dritte', 'zweite', 'zweiter', 'bitte',
            'hinsehen', 'stillen', 'genauerem', 'populär', 'ausgleichs', 'eindrucksvolle', 'prüf', 'auszufüllen',
            'wachsender', 'breiteren', 'mitzuwirken', 'traf', 'addieren', 'nährt', 'aufdrängen', 'vertagt',
            'auszutragen', 'vorzeitiger', 'besiegen', 'tagaus', 'tonner', 'rung', 'chen', 'däubler', 'gmelin',
            'schen', 'lich', 'merk', 'casper', 'sachkundiger', 'loyal', 'siebenmal', 'flegt', 'meisten', 'wert',
            'häufig', 'gelten', 'vieler', 'einzelne', 'besseren', 'zusammenbinden', 'modifizierte', 'inkompetent',
            'reinigen', 'saulus', 'paulus', 'inferno', 'anfallender', 'verantwortungsloses', 'aufmerksamer',
            'mußten', 'spüren', 'entschlossen', 'allmählich', 'bestätigen', 'zweitens', 'erstens', 'drittens',
            'viertens', 'fünftens', 'sechstens', 'siebtens', 'achtens', 'neuntens', 'dreierlei', 'verhielten',
            'eingemischt', 'kayser', 'timmermann', 'voßbeck', 'charles', 'wegen', 'erneut', 'schwach', 'haltet',
            'benutze', 'gepflegten', 'mauz', 'widmann', 'damerau', 'ratjen', 'sierra', 'leone', 'unvergessen',
            'anvisierten', 'einzuberufen', 'vertane', 'begonnen', 'füge', 'entstanden', 'beantwortet', 'duldet',
            'auszu', 'größer', 'täglich', 'sichtbar', 'mehreren', 'neuer', 'begreifen', 'zurückführen', 'deshalb',
            'dabei', 'brauchen', 'letzten', 'wurde', 'neue', 'unserer', 'schaffen', 'entschuldigen', 'vermuten',
            'hielt', 'viele', 'kommt', 'stellt', 'teil', 'eben', 'wichtige', 'erreichen', 'führen', 'zeigt',
            'gleichzeitig', 'worden', 'liegt', 'denen', 'abgesehen', 'ausweichen', 'stopfen', 'wahre', 'begangen',
            'letzteres', 'davon', 'fast', 'bekommen', 'ständig', 'genug', 'handelte', 'dorthin', 'vermag',
            'erschweren', 'insgesamt', 'hören', 'denken', 'betrieben', 'entsprechende', 'sogenannten', 'wenigen',
            'weitgehend', 'weder', 'hinsichtlich', 'dennoch', 'künftig', 'gewinnen', 'beraten', 'bloße', 'höchstem',
            'nebenbei', 'zugemutet', 'allemal', 'verzichtet', 'müsse', 'unterschiedlichen', 'entschieden', 'gezogen',
            'ebenfalls', 'wenigstens', 'jemand', 'vernachlässigen', 'tiefer', 'begreift', 'verändert', 'bislang',
            'schwierige','grundsätzlich', 'verstanden', 'solle', 'konkret', 'vollem', 'gedauert', 'selten',
            'sogenannte', 'außerordentlich', 'wohin', 'gelöst', 'gewaltige', 'streichen', 'vorgestern', 'kennt',
            'weist', 'gestärkt', 'funktioniert', 'eigener', 'zitiert', 'sorgt', 'lachen', 'zurückgegangen', 'kritisch',
            'dergleichen', 'wesentliches', 'fünftens', 'besonderen', 'sechstens', 'weitergehen', 'bemühen', 'einziges',
            'hingegen', 'angemessen', 'vornherein', 'unterschätzt', 'vermutlich']))

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


def filter_columns(df):
    init_speeches = df.to_dict('records')
    result = []
    for speech in init_speeches:
        if isinstance(speech['Speaker'], str) and 'CDU/CSU' in speech['Speaker']:
            speech['Speaker'] = ''
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
    output_path = 'data/preprocessed_full_sample/'
    final_speech_contribution = {}

    for filename in os.listdir(path):
        needed_speeches = 2500
        file = os.path.join(path, filename)
        bundestag = pd.read_csv(file)
        bundestag = bundestag.apply(
            lambda col: replace_special_characters(col) if col.name == "Speech text"
            else col)
        filtered = filter_columns(bundestag)
        df = pd.DataFrame(filtered)
        df_elements = df.sample(n=needed_speeches)
        print('SIZE: ', len(df_elements.index))
        final_speech_contribution[filename] = df_elements.apply(
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
