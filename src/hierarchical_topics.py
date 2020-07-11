import random
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from sklearn.manifold import TSNE
import scipy.sparse as ss

def print_all_topics(topic_model, filename, anchor_strength=0):
    with open(filename, "a+") as file:
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            file.write("With anchor strength " + str(anchor_strength) + " for topic 0 with the anchor"
                                                                        "word Kohle")
            topic_words, _ = zip(*topic)
            topic_words = [str(word) for word in topic_words]
            file.write('{}: '.format(n) + ','.join(topic_words) + "\n")


def predict_for_party(layers, corpus, vocabulary):
    with open("fdp_prediction.txt", "w+") as out:
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        document_term_matrix = vectorizer.fit_transform(corpus)

        for i, layer in enumerate(layers):
            out.write("Prediction of layer " + str(i) + ":\n\n\n")
            prediction_1 = layer.predict_proba(document_term_matrix)
            out.write(str(prediction_1))
            out.write("\n\n")
            document_term_matrix = prediction_1


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


def main():
    ninth_bundestag = pd.read_csv("data/bundestag_speeches_pp09-14/bundestag_speeches_pp9.csv_preprocessed.csv")
    speeches = ninth_bundestag["Speech text"]
    speeches = speeches.fillna("")
    speeches = speeches.tolist()
    coal_speeches = [speech for speech in speeches if "kohle" in speech]
    coal_speeches_train = coal_speeches[:100]
    test_speeches = coal_speeches[100:200]

    fdp_coal = ninth_bundestag.loc[ninth_bundestag["Speaker party"] == "fdp"]
    fdp_coal = fdp_coal["Speech text"]
    fdp_coal = fdp_coal.fillna("")
    fdp_coal = fdp_coal.tolist()
    fdp_coal = [speech for speech in fdp_coal if "kohle" in speech]

    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(coal_speeches_train).toarray()
    #convert matrix into sparse matrix, otherwise CorEx fails when used with anchors for some reason
    document_term_matrix = ss.csr_matrix(document_term_matrix)
    vocabs = vectorizer.get_feature_names()

    print("Begin topic extraction")

    for i in range(1, 5):

        topic_model = ct.Corex(n_hidden=5)
        topic_model.fit(document_term_matrix, words=vocabs, anchors=[["kohle"]], anchor_strength=i)

        print("First layer topics")
        visualize_topics(topic_model, test_speeches, vocabs)
        print_all_topics(topic_model, filename="OutLevel1.txt", anchor_strength=i)
        vt.vis_rep(topic_model, column_label=vocabs, prefix='topic-model-example')
    anchor_words = ['kernkraft', 'kernenergie', 'atomkraft']
    topic_model = ct.Corex(n_hidden=5, seed=2)
    topic_model.fit(document_term_matrix, words=vocabs, anchors=anchor_words, anchor_strength=6)

    predict_for_party([topic_model], fdp_coal, vocabs)
    tm_layer2 = ct.Corex(n_hidden=5, seed=2)
    tm_layer2.fit(topic_model.labels)

    print("Second layer topics")
    print_all_topics(tm_layer2, filename="OutLevel2.txt")
    vt.vis_rep(tm_layer2, column_label=list(map(str, range(80))), prefix='topic-model-example_layer2')

    tm_layer3 = ct.Corex(n_hidden=1, seed=2)
    tm_layer3.fit(tm_layer2.labels)

    print("Third layer topics")
    print_all_topics(tm_layer2, filename="OutLevel3.txt")
    vt.vis_rep(tm_layer3, column_label=list(map(str, range(8))), prefix='topic-model-example_layer3')

    vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=vocabs, max_edges=200)

    predict_for_party([topic_model, tm_layer2, tm_layer3], fdp_coal, vocabs)


if __name__ == "__main__":
    main()
