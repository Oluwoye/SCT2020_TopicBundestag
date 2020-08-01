import random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from sklearn.manifold import TSNE
import scipy.sparse as ss
import os

from tqdm import tqdm

from src.corex_topic_modeling import split_indices_per_party_and_seat_type, evaluate_corpus, predict_for_party, \
    predict_for_speaker


def print_all_topics(topic_model, filename, anchor_strength=0):
    with open(filename, "a+") as file:
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            file.write("With anchor strength " + str(anchor_strength) + " for topic 0 with the anchor"
                                                                        "word Kohle")
            topic_words, _ = zip(*topic)
            topic_words = [str(word) for word in topic_words]
            file.write('{}: '.format(n) + ','.join(topic_words) + "\n")


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


def split_indices_per_party(bundestag_frame):
    party_index = dict()
    parties = bundestag_frame["Speaker party"].unique()
    for party in parties:
        # We only consider lines with a clear party affiliation so we exclude nan values
        if type(party) != str:
            continue
        mask = bundestag_frame["Speaker party"] == party
        party_index[party] = [i for i, truth_value in enumerate(mask) if truth_value]

    return party_index


def split_indices_per_speaker(bundestag_frame):
    speaker_index = dict()
    speakers = bundestag_frame["Speaker"].unique()
    for speaker in speakers:
        # We only consider lines with a clear speaker so we exclude nan values
        if type(speaker) != str:
            continue
        mask = bundestag_frame["Speaker"] == speaker
        speaker_index[speaker] = [i for i, truth_value in enumerate(mask) if truth_value]

    return speaker_index


def predict_all(vocabs, topic_layers, bundestag_frame):
    corpus = bundestag_frame['Speech text']
    corpus = corpus.fillna("")
    output_path = "data" + os.path.sep + "output" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_file = output_path + "all" + "_prediction.txt"

    topic_ratios = evaluate_corpus(topic_layers, output_file, vocabs, corpus, print_matrices=False, return_ratios=True)
    return topic_ratios


def main():
    path = 'data/preprocessed_up_sample/'
    bundestag_frame = pd.DataFrame()
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if bundestag_frame.empty:
            bundestag_frame = pd.read_csv(file)
        else:
            bundestag_frame = pd.concat([bundestag_frame, pd.read_csv(file)], ignore_index=True)
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


if __name__ == "__main__":
    main()
