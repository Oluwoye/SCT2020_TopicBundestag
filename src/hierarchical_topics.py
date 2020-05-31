import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from corextopic import vis_topic as vt


def print_all_topics(topic_model, filename):
    with open(filename, "w+") as file:
        topics = topic_model.get_topics()
        for n, topic in enumerate(topics):
            topic_words, _ = zip(*topic)
            topic_words = [str(word) for word in topic_words]
            file.write('{}: '.format(n) + ','.join(topic_words) + "\n")


def predict_for_party(layers, corpus, vocabulary):
    with open("fdp_prediction.txt", "w+") as out:
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        document_term_matrix = vectorizer.fit_transform(corpus)

        for i, layer in enumerate(layers):
            out.write("Prediction of layer " + str(i) + ":\n\n\n")
            prediction_1 = layer.predict(document_term_matrix)
            out.write(str(prediction_1))
            out.write("\n\n")
            document_term_matrix = prediction_1


def main():
    ninth_bundestag = pd.read_csv("data/bundestag_speeches_pp09-14/bundestag_speeches_pp9.csv_preprocessed.csv")
    speeches = ninth_bundestag["Speech text"]
    speeches = speeches.fillna("")
    speeches = speeches.tolist()
    coal_speeches = [speech for speech in speeches if "kohle" in speech]

    fdp_coal = ninth_bundestag.loc[ninth_bundestag["Speaker party"] == "fdp"]
    fdp_coal = fdp_coal["Speech text"]
    fdp_coal = fdp_coal.fillna("")
    fdp_coal = fdp_coal.tolist()
    fdp_coal = [speech for speech in fdp_coal if "kohle" in speech]

    vectorizer = CountVectorizer()
    document_term_matrix = vectorizer.fit_transform(coal_speeches).toarray()
    vocabs = vectorizer.get_feature_names()

    print("Begin topic extraction")

    topic_model = ct.Corex(n_hidden=50)
    topic_model.fit(document_term_matrix, words=vocabs)

    print("First layer topics")
    print_all_topics(topic_model, filename="OutLevel1.txt")
    vt.vis_rep(topic_model, column_label=vocabs, prefix='topic-model-example')

    tm_layer2 = ct.Corex(n_hidden=5)
    tm_layer2.fit(topic_model.labels)

    print("Second layer topics")
    print_all_topics(tm_layer2, filename="OutLevel2.txt")
    vt.vis_rep(tm_layer2, column_label=list(map(str, range(50))), prefix='topic-model-example_layer2')

    tm_layer3 = ct.Corex(n_hidden=1)
    tm_layer3.fit(tm_layer2.labels)

    print("Third layer topics")
    print_all_topics(tm_layer2, filename="OutLevel3.txt")
    vt.vis_rep(tm_layer3, column_label=list(map(str, range(5))), prefix='topic-model-example_layer3')

    vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=vocabs, max_edges=200)

    predict_for_party([topic_model, tm_layer2, tm_layer3], fdp_coal, vocabs)


if __name__ == "__main__":
    main()
