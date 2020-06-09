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
    praes = ["Richard Stücklen", "Heinrich Windelen", "Annemarie Renger", "Georg Leber", "Richard Wurbs"]
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
