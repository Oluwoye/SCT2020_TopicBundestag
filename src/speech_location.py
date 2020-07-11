import pandas as pd


def has_number(speech):
    if '1' in speech or '2' in speech or '3' in speech or '4' in speech or '5' in speech or '6' in speech or '7' in speech or '8' in speech or '9' in speech or '0' in speech:
        return True
    else:
        return False


def main():
    ninth_bundestag = pd.read_csv("data/bundestag_speeches_pp09-14/bundestag_speeches_pp9.csv")
    print(ninth_bundestag.columns)
    # The document DB ID only refers to the sittings
    # document_ids = ninth_bundestag["Document DB ID"].to_list()
    # first_id = document_ids[0]
    # last_id = document_ids[-1]
    # first_indices = []
    # for i in range(first_id, last_id + 1):
    #      first_indices.append(document_ids.index(i))
    # document_db_ids_changes = ninth_bundestag.iloc[first_indices]

    speeches = ninth_bundestag["Speech text"].to_list()
    speakers = ninth_bundestag["Speaker"].to_list()
    # From: https://de.wikipedia.org/wiki/Pr%C3%A4sident_des_Deutschen_Bundestages
    praes = {
        '1': ['Dr. Erich Köhler', 'Dr. Hermann Ehlers', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Hermann Schäfer'],
        '': ['Dr. Hermann Ehlers', 'Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Ludwig Schneider', 'Dr. Max Becker (Hersfeld)'],
        '3': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Max Becker (Hersfeld)', 'Dr. Thomas Dehler', 'Dr. Victor-Emanuel Preusker'],
        '4': ['Dr. Eugen Gerstenmaier', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Erwin Schoettle', 'Dr. Thomas Dehler'],
        '5': ['Dr. Eugen Gerstenmaier', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Maria Probst', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Karl Mommer', 'Erwin Schoettle', 'Dr. Thomas Dehler', 'Walter Scheel'],
        '6': ['Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Carlo Schmid (Frankfurt)', 'Dr. Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        '7': ['Dr. Annemarie Renger', 'Kai-Uwe Hassel', 'Dr. Richard Jaeger', 'Dr. Hermann Schmitt-Vockenhausen', 'Liselotte Funcke'],
        '8': ['Dr. Karl Carstens (Fehmarn)', 'Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Dr. Annemarie Renger', 'Dr. Hermann Schmitt-Vockenhausen', 'Georg Leber', 'Liselotte Funck', 'Richard Wurbs'],
        '9': ['Richard Stücklen', 'Dr. Richard von Weizsäcker', 'Heinrich Windelen', 'Dr. Annemarie Renger', 'Georg Leber', 'Richard Wurbs'],
        '10': ['Dr. Rainer Barzel', 'Dr. Philipp Jenninger', 'Richard Stücklen', 'Dr. Annemarie Renger', 'Heinz Westphal', 'Richard Wurbs', 'Dieter-Julius Cronenberg (Arnsberg)'],
        '11': ['Dr. Philipp Jenninger', 'Dr. Rita Süssmuth', 'Richard Stücklen', 'Dr. Annemarie Renger', 'Heinz Westphal', 'Dieter-Julius Cronenberg (Arnsberg)'],
        '12': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Helmuth Becker (NienBerge)', 'Renate Schmidt (Nürnberg)', 'Dieter-Julius Cronenberg (Arnsberg)'],
        '13': ['Dr. Rita Süssmuth', 'Hans Klein (München)', 'Michaela Geiger', 'Hans-Ulrich Klose', 'Dr. Antje Vollmer', 'Dr. Burkhard Hirsch'],
        '14': ['Dr. h.c. Wolfgang Thierse', 'Dr. Rudolf Seiters', 'Anke Fuchs (Köln)', 'Petra Bläss', 'Dr. Antje Vollmer', 'Dr. Hermann Otto Solms'],
        '15': ['Dr. h.c. Wolfgang Thierse', 'Dr. Norbert Lammert', 'Dr. h.c. Susanne Kastner', 'Dr. Antje Vollmer', 'Dr. Hermann Otto Solms'],
        '16': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Dr. h.c. Susanne Kastner', 'Dr. h.c. Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        '17': ['Dr. Norbert Lammert', 'Gerda Hasselfeldt', 'Eduard Oswald', 'Dr. h.c. Wolfgang Thierse', 'Petra Pau', 'Katrin Göring-Eckhardt', 'Dr. Hermann Otto Solms'],
        '18': ['Dr. Norbert Lammert', 'Peter Hintze', 'Michaela Noll', 'Johannes Singhammer', 'Dr. h.c. Edelgard Bulmahn', 'Ulla Schmidt (Aachen)', 'Petra Pau', 'Claudia Roth (Augsburg)'],
        '19': ['Dr. Wolfgang Schäuble', 'Dr. Hans-Peter Friedrich (Hof)', 'Thomas Oppermann', 'Petra Pau', 'Claudia Roth (Augsburg)', 'Wolfgang Kubicki']
    }
    # for i, speech in enumerate(speeches):
    #     print(i)
    all_praes_speeches = [i for i, speech in enumerate(speeches) if type(speech) == str and speakers[i] in praes]
    print("Amount of all speeches by a president or vice-president of the 9th Bundestag: " + str(len(all_praes_speeches)))
    tagesordnung_speeches = [i for i, speech in enumerate(speeches) if type(speech) == str and speakers[i] in praes and
                             ("Punkt" in speech or "Zusatzpunkt" in speech or "Tagesordnung" in speech
                              or "rufe" in speech or "Drucksache" in speech)]
    print(
        "Amount of all speeches by a president or vice-president of the 9th Bundestag including Tagesordnungspunkt,"
        " Zusatzpunkt and Punkt: " + str(len(tagesordnung_speeches)))
    tagesordnung_general = ninth_bundestag.iloc[tagesordnung_speeches]
    tagesordnung_general.to_csv("all_proceedings")
    tagesordnung_index = [i for i, speech in enumerate(speeches) if type(speech) == str and
                          ("Punkt" in speech or "Zusatzpunkt" in speech or "Tagesordnung" in speech
                           or "rufe" in speech or "Drucksache" in speech) and
                          ("Klima" in speech or "Kohle" in speech or "Energie" in speech) and
                          has_number(speech) and speakers[i] in praes]
    tagesordnung_climate = ninth_bundestag.iloc[tagesordnung_index]
    tagesordnung_climate.to_csv("climate_proceedings.csv")
    print(tagesordnung_climate)


if __name__ == "__main__":
    main()
