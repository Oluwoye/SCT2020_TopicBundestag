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
    # TODO: check for duplicates
    return list(set(['ja', 'au', 'wa', 'nein', 'iii', 'sche', 'dy', 'ing', 'al', 'oh', 'frau', 'herr', 'kollege', 'ta',
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
            'hingegen', 'angemessen', 'vornherein', 'unterschätzt', 'vermutlich', 'lassen', 'gerade', 'deutlich',
            'gesagt', 'wäre', 'sollten', 'tagesordnungspunkt', 'beispiel', 'beim', 'gehen', 'allein', 'hinaus',
            'möchte', 'punkt', 'weise', 'erste', 'deren', 'jedoch', 'wesentlichen', 'erhebliche', 'hinzu',
            'offensichtlich', 'auseinanderfallen', 'dümmer', 'müßte', 'müßten', 'bewußt', 'bißchen', 'mußte',
            'aufgrund', 'lässt', 'bisschen', 'müssten', 'geehrte', 'außen', 'konnte', 'verantwortlich', 'länger',
            'wahr', 'bald', 'angekündigt', 'gefragt', 'offenbar', 'anstatt', 'vorderster', 'dummen', 'unglaubliche',
            'hineinkommen', 'probiert', 'leichtsinnig', 'herausheben', 'nachfolgenden', 'anfänglichen', 'anziehen',
            'alltäglich', 'einsetzbar', 'gefährlichsten', 'überschätzt', 'scheiterte', 'mitarbeiterinnen',
            'mitarbeitern', 'mitarbeiter', 'tiefgreifenden', 'beelzebub', 'elementare', 'abgehalten', 'unlauteren',
            'gelitten', 'aufarbeiten', 'einzigartigen', 'einzigartige', 'gebührt', 'schwerwiegende', 'kürzerer',
            'unerträgliche', 'verläßt', 'getragen', 'aufgenommen', 'geändert', 'befriedigende', 'höhlt', 'gebauten',
            'leidvollen', 'halbherzige', 'bloßes', 'auferlegte', 'zurückbleibt', 'dreiste', 'inntal', 'aquila',
            'gehtnichtmehr', 'ders', 'groden', 'kranich', 'fechter', 'waack', 'ühlingen', 'dierig', 'sütterlin',
            'malecha', 'minden', 'fibich', 'stattgegeben', 'erklären', 'spricht', 'geblieben', 'niemals', 'wann',
            'genossinnen', 'genossen', 'ropa', 'ungeheuerliche', 'denunzieren', 'geholfen', 'diktiert', 'kant',
            'bekomme', 'davonlaufen', 'umgekehrt', 'schließt', 'berührt', 'dauerhaften', 'größeren', 'ausgelöst',
            'sogenanntes', 'veranschlagt', 'zurückgeht', 'bewältigen', 'beklagen', 'hinterlassen', 'erkennbar']))


def get_speech_start_indicators():
    return ['das wort hat ', 'erteile ich das wort ', 'erteile ich dem ', 'erteile ich der ',
            'ich rufe die frage ', 'rufe ich die frage', 'rufe ich auch die frage',
            'ich erteile das wort', 'ich erteile ihm das wort', 'ich erteile ihm dazu das wort',
            'ich erteile ihnen das wort', 'ich erteile jetzt']


def get_tagesordnung_indicators():
    return [' rufe den punkt', ' zusatzpunkt ', ' der tagesordnung ', ' heutigen tagesordnung ',
            'zur tagesordnung', ' drucksache ']


def get_filter_indicators():
    return list(set(['kohle', 'kernkraft', 'energie', 'umwelt', 'klimawandel', 'erderwärmung',
            'ökologisch', 'abgase', 'abholzung', 'artenschutz', 'erdatmosphäre', 'bergbau',
            'bevölkerungswachstum', 'biogas', 'umweltministerium', 'chemikalien', 'verseuchung', 'co2',
            'dürre', 'naturkatastrophe', 'flut', 'elektromobilität', 'emission', 'erdgas',
            'erneuerbar', 'extremwetter', 'feinstaub', 'fischerei', 'fleischkonsum',
            'fossile', 'geothermie', 'gletscher', 'hitzewelle', 'klimaschutz', 'klimaskeptiker',
            'lichtverschmutzung', 'luftqualität', 'luftverschmutzung', 'mikroplastik', 'nachhaltigkeit',
            'nachwachsende rohstoffe', 'naturbewusstsein', 'naturschutz', 'ökobilanz', 'öl', 'ozon',
            'permafrost', 'photovoltaik', 'radioaktiv', 'recycling', 'regenwald', 'ressourcenschonung',
            'schadstoffe', 'smog', 'solar', 'strom', 'tschernobyl', 'überfischung', 'umweltpolitik',
            'umweltverschmutzung', 'klimagipfel', 'versiegelung', 'dürre', 'klimaforschung', 'natur',
            'fckw', 'dekarbonisierung', 'klimakrise', 'aerosole', 'albedo', 'anthropogen', 'barrel',
            'diesel', 'co2', 'emission', 'fracking', 'kernschmelze', 'kilowattstunde', 'klimafinanzierung',
            'kohleimporte', 'kohlendioxid', 'kokskohle', 'kyotoprotokoll', 'luftverkehr', 'meeresspiegel',
            'methan', 'montrealprotokoll', 'opec', 'permafrost', 'fotovoltaik', 'rspo', 'solarthermie',
            'stromkonzerne', 'stromexport', 'treibhauseffekt', 'treibhausgase', 'versauerung', 'wälder',
            'wärmepumpe', 'wärmestrahlung', 'weltklimarat', 'windenergie', 'cdm', 'ipcc', 'kohlenstoffsenke',
            'mitigation', 'napa', 'redd', 'thg', 'unfcc', 'cop', 'desertifikation', 'energielobby', 'kerosin',
            'klimatologie', 'stickstoffoxide', 'troposphäre', 'zivilisationskatastrophe', 'atomausstieg',
            'atomkraft', 'atomenergie', 'atommüll', 'biodiesel', 'bioenergie', 'bioethanol', 'biogas',
            'blockheizkraftwerke', 'brennstoffzelle', 'eigenerzeugung', 'einspeisevergütung', 'energieeffizienz',
            'energiestandard', 'erdwärme', 'fernwärmevorranggebiet', 'heizwert', 'windpark', 'ökostrom', 'smog',
            'wasserkraft', 'fußabdruck', 'klimapolitik', 'energiewende', 'atomausstieg', 'kohleausstieg',
            'klimaflüchtlinge', 'klimaleugner', 'hambach', 'eeg']))


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
            if tagged[0][1] in ['NN', 'NNS']:
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
    return mdb_names


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = word[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
