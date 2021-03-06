import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from sklearn.manifold import TSNE
import scipy.sparse as ss
import os

from tqdm import tqdm

from corex_topic_modeling import split_indices_per_party_and_seat_type, evaluate_corpus, predict_for_party, \
    predict_for_speaker, evaluate_document_topic_matrix, split_indices_per_party, split_indices_per_speaker


def print_all_topics(topic_model, filename, anchor_strength=0):
    with open(filename, "a+") as file:
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            file.write("With anchor strength " + str(anchor_strength) + " for topic 0 with the anchor"
                                                                        "word Kohle")
            topic_words, _ = zip(*topic)
            topic_words = [str(word) for word in topic_words]
            file.write('{}: '.format(n) + ','.join(topic_words) + "\n")


def autolabel_bars(rects, ax):
    for rect in rects:
        height = rect.get_pos()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def get_top_words_of_topics(d=',', n=3):
    with open("tm-example_layer1/topics.txt", 'r', encoding='utf_8') as topic_file:
        x_labels = topic_file.readlines()
        x_labels = [d.join(label.split(d, n)[:n]) for label in x_labels]
        x_labels = [label.replace('~', '') for label in x_labels]
    return x_labels


def visualize_topics(topic_model, speeches, vocabulary):
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    document_term_matrix = vectorizer.fit_transform(speeches)
    topics = np.array(topic_model.predict_proba(document_term_matrix))

    labels = []
    for topic in topics:
        labels.append(np.argmax(topic))

    perplexity_values = list(range(5, 50))
    learning_rate = list(range(10, 1000))
    early_exaggeration = list(range(5, 30))
    method = ['barnes_hut']
    n_iter = list(range(250, 2000))
    min_grad_norm = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

    for i in range(0, 5):
        tsne_params = {'n_components': 2, 'random_state': 42, 'perplexity': random.choice(perplexity_values),
                       'learning_rate': random.choice(learning_rate),
                       'early_exaggeration': random.choice(early_exaggeration), 'method': random.choice(method),
                       'n_iter': random.choice(n_iter), 'min_grad_norm': random.choice(min_grad_norm)}
        tsne_estimator = TSNE(n_components=tsne_params['n_components'], random_state=tsne_params['random_state'],
                              perplexity=tsne_params['perplexity'], learning_rate=tsne_params['learning_rate'],
                              early_exaggeration=tsne_params['early_exaggeration'], method=tsne_params['method'],
                              n_iter=tsne_params['n_iter'], min_grad_norm=tsne_params['min_grad_norm'])
        points = tsne_estimator.fit_transform(topics)

        points = np.array(points)
        annotations = list(range(len(points)))
        points = np.transpose(points)
        print(points)
        plt.scatter(points[0], points[1], c=labels)
        plt.title("Example embedding of CorEx_model per document")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        # for j, annotation in enumerate(annotations):
        #     plt.annotate(j, (points[0][j], points[1][j]))
        plt.tight_layout()
        plt.savefig("embedded_data_" + str(i) + ".png", bbox_inches='tight')
        plt.close()
    print(topics)


def split_indices_per_legislation_party(bundestag_frame, dates, parties_per_legislation):
    legislation_party_index = dict()
    for key, value in dates.items():
        for party in parties_per_legislation[key]:
            if type(party) != str:
                continue
            print(key, parties_per_legislation[key])
            print(party)
            name = key + '-' +  party
            mask = (bundestag_frame["Date"] >= value['start']) & (bundestag_frame["Date"] <= value["end"]) & (bundestag_frame["Speaker party"] == party)
            legislation_party_index[name] = [i for i, truth_value in enumerate(mask) if truth_value]

    return legislation_party_index


def query_corpus_for_speaker(bundestag_frame, indices_per_speaker, party_dict, speaker):
    speaker_indices = indices_per_speaker[speaker]
    party_values = None
    if party_dict is not None:
        party_affiliation = (bundestag_frame.loc[bundestag_frame['Speaker'] == speaker]['Speaker party']).tolist()[0]
        if party_affiliation in party_dict.keys():
            party_values = party_dict[party_affiliation]
        else:
            party_values = None
    speeches = bundestag_frame['Speech text']
    speeches = speeches.fillna("")
    corpus = speeches[speaker_indices]
    return corpus, party_values


def predict_all(vocabs, topic_layers, bundestag_frame):
    corpus = bundestag_frame['Speech text']
    corpus = corpus.fillna("")
    output_path = "data" + os.path.sep + "output" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_file = output_path + "all" + "_prediction.txt"

    topic_ratios = evaluate_corpus(topic_layers, output_file, vocabs, corpus, print_matrices=False, return_ratios=True)
    return topic_ratios


def compare_plot_per_topic(speaker_collection, general_ratio, indices_per_speaker, party_dict, topic_num, frame, vocabs,
                           layers, vocabulary):
    speaker_ratios = dict()
    party_ratios = dict()
    party_colours = {"gruene": "green", "spd": "red", "cducsu": "black", "fdp": "yellow", "linke": "darkred"}
    for key, values in speaker_collection.items():
        party_colour = party_colours[key]
        party_ratios[party_colour] = party_dict[key][topic_num]
        for speaker in values:
            corpus, ignored = query_corpus_for_speaker(frame, indices_per_speaker, party_dict, speaker)
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            document_term_matrix = vectorizer.fit_transform(corpus)
            prediction_1, ignored = layers[0].predict_proba(document_term_matrix)
            ignored, ratios = evaluate_document_topic_matrix(prediction_1, print_matrices=False,
                                                             return_ratios=True, general_entity=None,
                                                             comparison_entity=speaker)
            speaker_ratios[(speaker + " (" + key + ")")] = ratios[topic_num]

    arts = plot_politicians_results(speaker_ratios, general_ratio, party_ratios, get_top_words_of_topics(n=5)[topic_num])
    save_path = os.path.join("data", "plots", "comparison")
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "comparison_topic_" + str(topic_num) + ".png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_politicians_results(scores, general_ratio, party_ratios, topic):
    #  create the figure
    fig, ax1 = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(left=0.115, right=0.88)
    fig.canvas.set_window_title('Comparison for several speakers')

    pos = np.arange(len(scores.keys()))
    keys = []
    values = []
    for key, value in scores.items():
        keys.append(key)
        values.append(value)
    place_holders = [''] * len(keys)

    rects = ax1.barh(pos, values,
                     align='center',
                     height=0.5,
                     alpha=0.25,
                     tick_label=place_holders
                     )

    ax1.set_title("Comparison for topic: " + topic + " over several politicians")

    # ax1.set_xlim([0, 1])
    # ax1.xaxis.set_major_locator(MaxNLocator(11))
    # ax1.xaxis.grid(True, linestyle='--', which='major',
    #               color='grey', alpha=.25)

    # Plot a solid vertical gridline to highlight the median position
    ax1.axvline(general_ratio, color='grey', alpha=0.25)
    for party_colour, party_value in party_ratios.items():
        ax1.axvline(party_value, color=party_colour, alpha=0.25)

    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()

    # set the tick locations
    ax2.set_yticks(pos)
    # make sure that the limits are set equally on both yaxis so the
    # ticks line up
    ax2.set_ylim(ax1.get_ylim())

    # set the tick labels
    ax2.set_yticklabels(scores.keys())

    ax2.set_ylabel('Politician')

    ax1.set_xlabel("Share of speeches regarding the topic")

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for i in range(len(rects)):
        rect = rects[i]
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = str(values[i])
        # The bars aren't wide enough to print the ranking inside
        if width < 0.4:
            # Shift the text to the right side of the right edge
            xloc = 5
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -5
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                             textcoords="offset points",
                             ha=align, va='center',
                             color=clr, weight='bold', clip_on=True)
        rect_labels.append(label)

    # return all of the artists created
    return {'fig': fig,
            'ax': ax1,
            'ax_right': ax2,
            'bars': rects,
            'perc_labels': rect_labels}


def main():
    path = 'data/preprocessed_up_sample/'
    bundestag_frame = pd.DataFrame()
    legislation_dates = {}
    parties_per_legislation = {}
    for filename in os.listdir(path):
        if filename == 'bundestag_19.csv':
            print('19 filtered')
            continue
        file = os.path.join(path, filename)
        new_data = pd.read_csv(file)
        legislation_dates[filename] = {}
        print(filename)
        print(new_data['Date'].min())
        print(new_data['Date'].max())
        print(new_data['Speaker party'].unique().tolist())
        print('--------------------------------')
        legislation_dates[filename]['start'] = new_data['Date'].min()
        legislation_dates[filename]['end'] = new_data['Date'].max()
        parties_per_legislation[filename] = new_data['Speaker party'].unique().tolist()
        if bundestag_frame.empty:
            bundestag_frame = new_data
        else:
            bundestag_frame = pd.concat([bundestag_frame, new_data], ignore_index=True)
    # bundestag_frame = pd.read_csv("data/merged/final_single/newbundestag_speeches_pp17.csv")
    indices_per_party = split_indices_per_party(bundestag_frame)
    indices_per_speaker = split_indices_per_speaker(bundestag_frame)
    seats_frame = pd.read_csv("data/seats.csv")
    people_frame = pd.read_csv("data/people.csv")
    people_frame.rename(columns={"id": "occupant__id"}, inplace=True)
    merged_frame = people_frame.merge(seats_frame, on=["occupant__id"])
    merged_frame = merged_frame.loc[merged_frame["clean_name"].isin(indices_per_speaker.keys())]
    merged_frame = merged_frame[["clean_name", "seat_type"]]
    merged_frame.rename(columns={"clean_name": "Speaker"}, inplace=True)
    merged_frame = merged_frame.merge(bundestag_frame, on=["Speaker"])
    indices_per_party_and_seat_type = split_indices_per_party_and_seat_type(merged_frame)
    # bundestag_frame = bundestag
    indices_per_legislation_party = split_indices_per_legislation_party(bundestag_frame, legislation_dates, parties_per_legislation)
    print('INDIZES')
    bad_keys = [key for key, val in indices_per_legislation_party.items() if len(val) == 0]
    for key in bad_keys:
        del indices_per_legislation_party[key]
    speeches = bundestag_frame["Speech text"]
    speeches = speeches.fillna("")
    speeches = speeches.tolist()
    # coal_speeches = [speech for speech in speeches if "kohle" in speech]

    # fdp_coal = bundestag_frame.loc[bundestag_frame["Speaker party"] == "fdp"]
    # fdp_coal = fdp_coal["Speech text"]
    # fdp_coal = fdp_coal.fillna("")
    # fdp_coal = fdp_coal.tolist()
    # fdp_coal = [speech for speech in fdp_coal if "kohle" in speech]

    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(speeches).toarray()
    # convert matrix into sparse matrix, otherwise CorEx fails when used with anchors for some reason
    document_term_matrix = ss.csr_matrix(document_term_matrix)
    vocabs = vectorizer.get_feature_names()

    print("Begin topic extraction")

    #    for i in range(1, 5):

    #        topic_model = ct.Corex(n_hidden=5)
    #       topic_model.fit(document_term_matrix, words=vocabs, anchors=[["kohle"]], anchor_strength=i)
    #
    #       print("First layer topics")
    #        visualize_topics(topic_model, coal_speeches, vocabs)
    #        print_all_topics(topic_model, filename="OutLevel1.txt", anchor_strength=i)
    #        vt.vis_rep(topic_model, column_label=vocabs, prefix='topic-model-example')
    anchor_words = [['kohle', 'bergbau'], ['kernenergie', 'atomkraft'], ['solarenergie', 'wasserkraft']]
    topic_model = ct.Corex(n_hidden=50, seed=2)
    topic_model.fit(document_term_matrix, words=vocabs)
    print('tc', topic_model.tc)
    for idx, val in enumerate(topic_model.tcs):
        print('topic ', idx, ' tc: ', val)
    print_all_topics(topic_model, filename="level1.txt")
    vt.vis_rep(topic_model, column_label=vocabs, prefix='tm-example_layer1')

    tm_layer2 = ct.Corex(n_hidden=5, seed=2)
    tm_layer2.fit(topic_model.labels)
    print('tc 2nd', tm_layer2.tc)
    for idx, val in enumerate(tm_layer2.tcs):
        print('topic ', idx, ' tc: ', val)
    print("Second layer topics")
    print_all_topics(tm_layer2, filename="Level2.txt")
    vt.vis_rep(tm_layer2, column_label=list(map(str, range(50))), prefix='tm-example_layer2')

    tm_layer3 = ct.Corex(n_hidden=1, seed=2)
    tm_layer3.fit(tm_layer2.labels)

    print("Third layer topics")
    print_all_topics(tm_layer2, filename="Level3.txt")
    vt.vis_rep(tm_layer3, column_label=list(map(str, range(5))), prefix='tm-example_layer3')

    vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=vocabs, max_edges=200)

    general_ratios = predict_all(vocabs, [topic_model, tm_layer2, tm_layer3], bundestag_frame)
    legislation_ratios = {}
    for key, val in legislation_dates.items():
        legislation_frame = bundestag_frame.loc[(bundestag_frame["Date"] >= val["start"]) & (bundestag_frame["Date"] <= val["end"])]
        legislation_ratios[key] = predict_all(vocabs, [topic_model, tm_layer2, tm_layer3], legislation_frame)
    parties = list(indices_per_party.keys())
    party_dict = dict()
    for i in tqdm(range(0, len(parties))):
        party_dict = predict_for_party(indices_per_party, vocabs, [topic_model, tm_layer2, tm_layer3], parties[i],
                                       bundestag_frame,
                                       general_entity=general_ratios, party_dict=party_dict,
                                       party_dict_with_seat_type=indices_per_party_and_seat_type)
    legislation_parties = list(indices_per_legislation_party.keys())
    for i in tqdm(range(0, len(legislation_parties))):
        predict_for_speaker(indices_per_legislation_party, vocabs, [topic_model, tm_layer2, tm_layer3],
                                legislation_parties[i], bundestag_frame,
                                general_entity=legislation_ratios[legislation_parties[i].split('-')[0]],
                                party_dict=party_dict, do_it_special=True)
    # for i in tqdm(range(0, len(legislation_parties))):
    #     legislation_party_dict = predict_for_party(indices_per_legislation_party, vocabs, [topic_model, tm_layer2, tm_layer3],
    #                                                 legislation_parties[i], bundestag_frame, general_entity=general_ratios, party_dict=legislation_party_dict)
    speakers = list(indices_per_speaker.keys())
    for i in tqdm(range(0, len(speakers))):
        # This speaker has a question mark at the end of his name after preprocessing. Therefore we exclude him.
        if "Wolfgang Ne" in speakers[i]:
            continue
        predict_for_speaker(indices_per_speaker, vocabs, [topic_model, tm_layer2, tm_layer3], speakers[i],
                            bundestag_frame, general_entity=general_ratios, party_dict=party_dict)

    speaker_collection = dict()
    speaker_collection["cducsu"] = ["Dr. Angela Merkel"]
    speaker_collection["gruene"] = ["Renate Künast"]
    speaker_collection["fdp"] = ["Michael Kauch"]
    speaker_collection["linke"] = ["Jan Aken"]
    speaker_collection["spd"] = ["Sigmar Gabriel"]

    for i in range(50):
        compare_plot_per_topic(speaker_collection, general_ratios[i], indices_per_speaker, party_dict, i,
                               bundestag_frame, vocabs, [topic_model], vocabs)


if __name__ == "__main__":
    main()
