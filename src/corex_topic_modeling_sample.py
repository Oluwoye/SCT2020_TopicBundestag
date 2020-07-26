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


def print_all_topics(topic_model, filename, anchor_strength=0):
    with open(filename, "a+") as file:
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            file.write("With anchor strength " + str(anchor_strength) + " for topic 0 with the anchor"
                                                                        "word Kohle")
            topic_words, _ = zip(*topic)
            topic_words = [str(word) for word in topic_words]
            file.write('{}: '.format(n) + ','.join(topic_words) + "\n")


def plot_topic_ratios(plottable_elements, comparison_entity, general_entity=None, party_values=None):
    y_pos = np.arange(len(plottable_elements))
    plt.bar(y_pos, plottable_elements, align='center', alpha=0.5)

    if general_entity is not None:
        y_pos = np.arange(len(general_entity))
        plt.bar(y_pos, general_entity, align='center', alpha=0.5)

    if party_values is not None:
        y_pos = np.arange(len(general_entity))
        plt.bar(y_pos, party_values, align='center', alpha=0.5)

    plt.ylabel('Ratio of topic assignments')
    plt.title('Topic distribution for all documents vs. ' + comparison_entity + ' documents')
    plt.legend([comparison_entity + " topic ratios", "overall topic ratios", "values for according party"])
    plt.tight_layout()

    output_path = "data" + os.path.sep + "plots" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    plt.savefig(output_path + comparison_entity + "_barplot.png")
    plt.close()

    # Piechart seems not to fit the use-case
    # labels = []
    # for i in range(0, len(plottable_elements)):
    #    labels.append("topic " + str(i))
    # patches, texts = plt.pie(plottable_elements, shadow=True, startangle=90)
    # plt.legend(patches, labels, loc="best")
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(output_path + comparison_entity + "_pieplot.png")
    # plt.close()


def evaluate_document_topic_matrix(document_topic_matrix, comparison_entity=None, general_entity=None,
                                   print_matrices=True,
                                   return_ratios=False, party_values=None):
    document_topic_matrix = np.array(document_topic_matrix)
    res_string = ""

    mean_per_topic = np.mean(document_topic_matrix, axis=0)
    topic_occurences = np.count_nonzero(document_topic_matrix >= 0.7, axis=0)
    total_amount = np.sum(topic_occurences)
    average_topic_assignments = np.mean(np.count_nonzero(document_topic_matrix >= 0.7, axis=1))

    if general_entity is not None and comparison_entity is not None:
        plot_topic_ratios(topic_occurences / total_amount, comparison_entity, general_entity, party_values=party_values)

    if print_matrices:
        res_string += "\nOriginal document-topic matrix: \n"
        for row in document_topic_matrix:
            res_string += str(row) + "\n"
    res_string += "Shape of document-topic matrix: \n" + str(document_topic_matrix.shape) + "\n"
    res_string += "Total number of elements: \n" + str(total_amount) + "\n"
    res_string += "Mean confidence per topic: \n" + str(mean_per_topic) + "\n"
    res_string += "Number of occurences per topic with confidence over 0.7: \n" + str(topic_occurences) + "\n"
    res_string += "Ratios of occurences per topic with confidence over 0.7: \n" + str(
        topic_occurences / total_amount) + "\n"
    res_string += "Average topic assignments per document with confidence over 0.7: \n" + str(
        average_topic_assignments) + "\n"

    if return_ratios:
        return res_string, topic_occurences / total_amount
    else:
        return res_string


def predict_for_party(party_index_dict, vocabulary, layers, party, bundestag_dataframe, general_entity=None,
                      party_dict=None):
    party_indices = party_index_dict[party]
    speeches = bundestag_dataframe['Speech text']
    speeches = speeches.fillna("")
    corpus = speeches[party_indices]
    output_path = "data" + os.path.sep + "output" + os.path.sep + "parties" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_file = output_path + party + "_prediction.txt"

    party_dict = evaluate_corpus(layers, output_file, vocabulary, corpus, print_matrices=False,
                                 general_entity=general_entity,
                                 comparison_entity=party, party_dict=party_dict)
    return party_dict


def evaluate_corpus(layers, output_file, vocabulary, corpus, print_matrices=True, return_ratios=False,
                    general_entity=None,
                    comparison_entity=None, party_dict=None, party_values=None):
    with open(output_file, "w+") as out:
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        document_term_matrix = vectorizer.fit_transform(corpus)
        topic_ratios = []

        for i, layer in enumerate(layers):
            out.write("Prediction of layer " + str(i) + ":\n\n\n")
            prediction_1, ignored = layer.predict_proba(document_term_matrix)
            if party_dict is not None and i == 0:
                evaluation, ratios = evaluate_document_topic_matrix(prediction_1, print_matrices=print_matrices,
                                                                    return_ratios=True, general_entity=general_entity,
                                                                    comparison_entity=comparison_entity)
                party_dict[comparison_entity] = ratios
            else:
                evaluation = evaluate_document_topic_matrix(prediction_1, print_matrices=print_matrices,
                                                            return_ratios=return_ratios, general_entity=general_entity,
                                                            comparison_entity=comparison_entity,
                                                            party_values=party_values)

            if return_ratios and i == 0:
                topic_ratios = evaluation[1]
                evaluation = evaluation[0]
            elif return_ratios:
                evaluation = evaluation[0]
            out.write(evaluation)
            out.write("\n\n")
            document_term_matrix = np.array(prediction_1)
            general_entity = None

        if return_ratios:
            return topic_ratios
        elif party_dict is not None:
            return party_dict


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


def predict_for_speaker(indices_per_speaker, vocabs, topic_layers, speaker, bundestag_frame, general_entity=None,
                        party_dict=None):
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
    output_path = "data" + os.path.sep + "output" + os.path.sep + "speakers" + os.path.sep
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    output_file = output_path + speaker + "_prediction.txt"

    evaluate_corpus(topic_layers, output_file, vocabs, corpus, print_matrices=False, general_entity=general_entity,
                    comparison_entity=speaker, party_values=party_values)


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
                                       general_entity=general_ratios, party_dict=party_dict)
    speakers = list(indices_per_speaker.keys())
    for i in tqdm(range(0, len(speakers))):
        # This speaker has a question mark at the end of his name after preprocessing. Therefore we exclude him.
        if "Wolfgang Ne" in speakers[i]:
            continue
        predict_for_speaker(indices_per_speaker, vocabs, [topic_model, tm_layer2, tm_layer3], speakers[i],
                            bundestag_frame, general_entity=general_ratios, party_dict=party_dict)


if __name__ == "__main__":
    main()
