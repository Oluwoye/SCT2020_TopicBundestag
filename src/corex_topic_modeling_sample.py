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

from src.corex_topic_modeling import evaluate_corpus, split_indices_per_party, split_indices_per_speaker, \
    split_indices_per_party_and_seat_type, predict_for_party, predict_for_speaker, evaluate_document_topic_matrix


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


def plot_topic_ratios(plottable_elements, comparison_entity, general_entity=None, party_values=None,
                      suffix=None):
    y_pos = (np.arange(len(plottable_elements)))
    fig, ax = plt.subplots()
    ax.scatter(y_pos, plottable_elements, s=20, alpha=1.0)

    if general_entity is not None:
        ax.scatter(y_pos, general_entity, s=20, alpha=1.0)
        # for xy in zip(y_pos, general_entity):
        #     ax.annotate('%s' % xy[1], xy=xy, textcoords='data')

    if party_values is not None:
        ax.scatter(y_pos, party_values, s=20, alpha=1.0)

    ax.legend([comparison_entity + " topic ratios", "overall topic ratios", "values for according party"])

    for i in range(50):
        ax.axvline(i, color='grey', alpha=0.1)

    ax.set_ylabel('Ratio of topic assignments')
    ax.set_title('Topic distribution for all documents vs. ' + comparison_entity + ' documents')
    x_labels = get_top_words_of_topics()
    ax.set_xticks(y_pos)
    ax.set_xticklabels(x_labels, rotation=90)
    fig.subplots_adjust(left=0.005, right=1. - 0.005)

    # fig.subplots_adjust(bottom=0.3)
    # fig.tight_layout()

    output_path = "data" + os.path.sep + "plots" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if suffix is not None:
        fig.savefig(output_path + comparison_entity + "_barplot_" + suffix + ".png", bbox_inches='tight')
    else:
        fig.savefig(output_path + comparison_entity + "_barplot.png", bbox_inches='tight')
    plt.close()


def get_top_words_of_topics(d=',', n=3):
    with open("tm-example_layer1/topics.txt", 'r', encoding='utf_8') as topic_file:
        x_labels = topic_file.readlines()
        x_labels = [d.join(label.split(d, n)[:n]) for label in x_labels]
        x_labels = [label.replace('~', '') for label in x_labels]
    return x_labels


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
    path = 'data/preprocessed_full_sample/'
    bundestag_frame = pd.DataFrame()
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if bundestag_frame.empty:
            bundestag_frame = pd.read_csv(file)
        else:
            bundestag_frame = pd.concat([bundestag_frame, pd.read_csv(file)], ignore_index=True)
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
    parties = list(indices_per_party.keys())
    party_dict = dict()
    for i in tqdm(range(0, len(parties))):
        party_dict = predict_for_party(indices_per_party, vocabs, [topic_model, tm_layer2, tm_layer3], parties[i],
                                       bundestag_frame,
                                       general_entity=general_ratios, party_dict=party_dict,
                                       party_dict_with_seat_type=indices_per_party_and_seat_type)
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
    speaker_collection["fdp"] = ["Christian Lindner"]
    speaker_collection["linke"] = ["Jan Aken"]
    speaker_collection["spd"] = ["Sigmar Gabriel"]

    for i in range(50):
        compare_plot_per_topic(speaker_collection, general_ratios[i], indices_per_speaker, party_dict, i,
                               bundestag_frame, vocabs, [topic_model], vocabs)


if __name__ == "__main__":
    main()
