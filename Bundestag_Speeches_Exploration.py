#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys


def main():
    # Don't use this in production
    explore_ninth_bundestag()
    explore_19th_bundestag()
    explore_mandate_data(since_parlperiod=9)


def explore_mandate_data(since_parlperiod=1, until_parperiod=19):
    constituencies = pd.read_csv("data/constituencies.csv")
    lists = pd.read_csv('data/lists.csv')
    people = pd.read_csv('data/people.csv')
    seats = pd.read_csv('data/seats.csv')

    print("MPs characteristic since " + str(since_parlperiod) + "period till " + str(until_parperiod) + " period.")

    joined = pd.merge(people, seats, left_on=["id"], right_on=["occupant__id"])
    direct_elected_mps_all = joined.loc[joined['seat_type'] == 1]
    direct_elected_mps = direct_elected_mps_all[
        (direct_elected_mps_all['parlperiod__n'] >= since_parlperiod) &
        (direct_elected_mps_all['parlperiod__n'] <= until_parperiod)]
    direct_elected_mps = direct_elected_mps['clean_name'].unique()
    direct_elected_mps_all = direct_elected_mps_all['clean_name'].unique()
    print("Direct elected mps")
    print(str(direct_elected_mps).encode('utf-8'))
    print("Number of distinct direct elected mps")
    print(len(direct_elected_mps))

    list_mps_all = joined.loc[joined['seat_type'] == 2]
    list_mps = list_mps_all[
        (list_mps_all['parlperiod__n'] >= since_parlperiod) & (list_mps_all['parlperiod__n'] <= until_parperiod)]
    list_mps = list_mps['clean_name'].unique()
    list_mps_all = list_mps_all['clean_name'].unique()
    print("List mps")
    print(str(list_mps).encode('utf-8'))
    print("Number of distinct list mps")
    print(len(list_mps))

    all_distinct_mps = np.union1d(direct_elected_mps, list_mps)
    print("All distinct mps")
    print(str(all_distinct_mps).encode('utf-8'))
    print("Number of all distinct mps")
    print(len(all_distinct_mps))

    mps_list_direct = np.intersect1d(direct_elected_mps, list_mps)
    print("All mps which held a list and a direct mandate")
    print(str(mps_list_direct).encode('utf-8'))
    print("Number of mps which held a list and a direct mandata")
    print(len(mps_list_direct))

    only_list_mps = np.setdiff1d(list_mps, direct_elected_mps_all)
    print("All mps with list mandate only")
    print(str(only_list_mps).encode('utf-8'))
    print("Number of mps with list mandate only")
    print(len(only_list_mps))

    only_direct_mps = np.setdiff1d(direct_elected_mps, list_mps_all)
    print("All mps with direct mandate only")
    print(str(only_direct_mps).encode('utf-8'))
    print("Number of mps with direct mandate only")
    print(len(only_direct_mps))


def explore_19th_bundestag():
    ninth_bundestag = pd.read_csv('data/bundestag_speeches_pp14-19/bundestag_speeches_pp19.csv')
    print(ninth_bundestag.columns)
    parties = ninth_bundestag['Speaker party'].unique()
    print("Parties in the 19th Bundestag:")
    print(parties)

    coal_lines = ninth_bundestag.loc[ninth_bundestag['Speech text'].str.contains('Kohle')]
    coal_lines = coal_lines[['Speech text']]
    coal_lines.to_csv('research_content/coal_mention_pp19.csv', header=True)
    print("Number of speeches mentioning coal in the 19th Bundestag")
    print(coal_lines.shape[0])
    climate_lines = ninth_bundestag.loc[ninth_bundestag['Speech text'].str.contains('Klima')]
    climate_lines = climate_lines[['Speech text']]
    print("Number of speeches mentioning climate in the 19th Bundestag")
    print(climate_lines.shape[0])
    # climate_lines.to_csv('research_content/climate_mention_pp19.csv', header=True)


def explore_ninth_bundestag():
    ninth_bundestag = pd.read_csv('data/bundestag_speeches_pp09-14/bundestag_speeches_pp9.csv')
    print(ninth_bundestag.columns)
    print("Number of lines total")
    print(ninth_bundestag.shape[0])
    parties = ninth_bundestag['Speaker party'].unique()
    print("Parties in the 9th Bundestag:")
    print(parties)
    speaker_parties = ninth_bundestag['Speaker party']
    wav_lines = ninth_bundestag.loc[ninth_bundestag['Speaker party'] == 'wav']
    print('Wav speakers')
    print(wav_lines['Speaker'].unique())
    kpd_lines = ninth_bundestag.loc[ninth_bundestag['Speaker party'] == 'kpd']
    print('Kpd speakers')
    print(kpd_lines['Speaker'].unique())
    print(kpd_lines['Speech text'])

    nan_lines = ninth_bundestag.loc[pd.isna(ninth_bundestag['Speaker party'])]
    print('Unknown speakers')
    print(nan_lines['Speaker'].unique())
    print(nan_lines['Speech text'])

    coal_lines = ninth_bundestag.loc[ninth_bundestag['Speech text'].str.contains('Kohle')]
    coal_lines = coal_lines[['Speech text']]
    print("Number of speeches mentioning coal in the 9th Bundestag")
    print(coal_lines.shape[0])
    # coal_lines.to_csv('research_content/coal_mention_pp9.csv', header=True)
    climate_lines = ninth_bundestag.loc[ninth_bundestag['Speech text'].str.contains('Klima')]
    climate_lines = climate_lines[['Speech text']]
    print("Number of speeches mentioning climate in the 9th Bundestag")
    print(climate_lines.shape[0])
    # climate_lines.to_csv('research_content/climate_mention_pp9.csv', header=True)


if __name__ == '__main__':
    main()
