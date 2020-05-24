'''
Topic Modeling
'''

import gensim
import nltk
import os
import pandas as pd

def LDA(texts, model_label, number_of_topics=40):
    mallet_path = "C:\\mallet\\bin\\mallet"

    corpus_dictionary = gensim.corpora.Dictionary(texts)
    # corpus_dictionary.filter_extremes(no_below=5, no_above=0.5)
    processed_corpus = [corpus_dictionary.doc2bow(text) for text in texts]

    lda_model = gensim.models.wrappers.ldamallet.LdaMallet(
        mallet_path,
        corpus=processed_corpus,
        id2word=corpus_dictionary,
        num_topics=number_of_topics,
        optimize_interval=10,
        prefix=model_label
    )

    lda_model.save(model_label + '.p')

    topics = lda_model.show_topics(num_topics=number_of_topics, num_words=10)
    print(model_label)
    for topic in topics:
        print(f"Topic#{topic[0]}: {topic[1]}...\n\n")
    print('------------')

def build_texts(data):
    result = {
        'full': {
            'texts': [],
            'labels': []
        },
        'parties': {}
    }
    #convert needed columns to lists
    all_speeches = data['Speech text'].tolist()
    all_ids = data['Speech DB ID'].tolist()
    all_speakers = data['Speaker'].tolist()
    all_speaker_parties = data['Speaker party'].tolist()
    #all_interjection = data['Interjection content'].tolist()
    filter_count = 0
    for idx, speech in enumerate(all_speeches):
        #skip faulty speeches
        #there are quite a lot of speeches being filtered out now
        #due to being empty, maybe have to recheck preprocessing
        if not isinstance(speech, str):
            filter_count += 1
            continue
        #assign belonging labels
        speech_id = str(all_ids[idx]) if all_ids[idx] else ''
        speech_speaker = all_speakers[idx] if isinstance(all_speakers[idx], str) else 'no_speaker'
        speech_party = all_speaker_parties[idx] if isinstance(all_speaker_parties[idx], str) else 'no_party'
        if speech_party not in result['parties']:
            result['parties'][speech_party] = {'texts': [], 'labels': []}
        tokens = nltk.word_tokenize(speech)
        #only keep actually clean words, probably don't need this anymore after preprocessing
        cleaned = [word for word in tokens if word.isalnum()]
        result['full']['texts'].append(cleaned)
        result['parties'][speech_party]['texts'].append(cleaned)

        result['full']['labels'].append(speech_id + ' ' + speech_speaker)
        result['parties'][speech_party]['labels'].append(speech_id + ' ' + speech_speaker)
    print('filtered_speeches_count: ', filter_count)
    return result

def main():
    path = 'data/preprocessed/'
    list_of_filters = ['kohle', 'umwelt', 'klima']

    for file_name in os.listdir(path):
        #only do analysis for Bundestag 9 for now
        # remove this to build topic models for all preprocessed files
        if file_name != 'bundestag_speeches_pp9.csv':
            continue
        file = os.path.join(path, file_name)
        #read data
        df = pd.read_csv(file)

        #build filtered data (or condition on list_of_filters)
        filtered_df = df[df['Speech text'].str.contains('|'.join(list_of_filters), na = False)]

        #build input texts for LDA
        complete_data = build_texts(df)
        filtered_data = build_texts(filtered_df)
        
        #LDA on complete and party-wise unfiltered data
        LDA(complete_data['full']['texts'], 'all_speeches_unfiltered')
        for key, value in complete_data['parties'].items():
            LDA(value['texts'], key + '_speeches_unfiltered')
        
        #LDA on complete and party-wise keyword filtered data
        LDA(filtered_data['full']['texts'], 'all_speeches_unfiltered')
        for key, value in filtered_data['parties'].items():
            LDA(value['texts'], key + '_speeches_unfiltered')

if __name__ == '__main__':
    main()