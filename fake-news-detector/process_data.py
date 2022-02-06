"""
    process_data.py - Edward Gorman - eg6g17@soton.ac.uk
"""
import re
import csv
import string
import pandas as pd
from datetime import datetime
import textblob
from nltk.corpus import wordnet as wn, stopwords
from nltk import word_tokenize, defaultdict, WordNetLemmatizer, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(file):
    df = pd.read_csv(file, sep='\t', quoting=csv.QUOTE_NONE, escapechar="\\", encoding="utf-8")
    df = df.drop_duplicates()
    df['label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)
    return df


def get_date(date):
    date = re.sub(r": ", ':', date)
    return datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y')


def extract_text_length(df):
    labels = df['label']
    df.drop(['tweetId', 'userId', 'imageId(s)', 'username', 'timestamp', 'label'], axis=1, inplace=True)
    df.insert(1, 'textlength', df['tweetText'].apply(lambda x: len(x)), True)
    df.drop(['tweetText'], axis=1, inplace=True)
    return df, labels


def extract_timestamp_feats(df):
    labels = df['label']
    df.drop(['tweetId', 'tweetText', 'userId', 'imageId(s)', 'username', 'label'], axis=1, inplace=True)
    df.insert(1, 'timeofday', df['timestamp'].apply(lambda x: get_date(x).hour), True)
    df.insert(1, 'dayofweek', df['timestamp'].apply(lambda x: get_date(x).weekday()), True)
    df.drop(['timestamp'], axis=1, inplace=True)
    return df, labels


def get_tfidf_vect(df):
    df.drop(['tweetId', 'userId', 'imageId(s)', 'username', 'timestamp', 'label'], axis=1, inplace=True)

    df['tweetText'].dropna(inplace=True)
    df['tweetText'] = [entry.lower() for entry in df['tweetText']]
    df['tweetText'] = df['tweetText'].apply(lambda x: re.sub(r"http[s]://(\w+)", '', x))
    df['tweetText'] = df['tweetText'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['tweetText'] = [word_tokenize(entry) for entry in df['tweetText']]
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index, entry in enumerate(df['tweetText']):
        final_words = []
        word_lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        df.loc[index, 'text_final'] = str(final_words)

    df = df.dropna()
    tfidf_vect = TfidfVectorizer(max_features=4000)
    tfidf_vect.fit(df['text_final'])
    return tfidf_vect


def extract_text_tfidf(df, tfidf_vect):
    df.drop(['tweetId', 'userId', 'imageId(s)', 'username', 'timestamp'], axis=1, inplace=True)

    df['tweetText'].dropna(inplace=True)
    df['tweetText'] = [entry.lower() for entry in df['tweetText']]
    df['tweetText'] = df['tweetText'].apply(lambda x: re.sub(r"http[s]://(\w+)", '', x))
    df['tweetText'] = df['tweetText'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['tweetText'] = [word_tokenize(entry) for entry in df['tweetText']]
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index, entry in enumerate(df['tweetText']):
        final_words = []
        word_lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_final = word_lemmatized.lemmatize(word, tag_map[tag[0]])
                final_words.append(word_final)
        df.loc[index, 'text_final'] = str(final_words)

    df = df.dropna()
    return tfidf_vect.transform(df['text_final']), df['label']


def check_pos_tag(x, flag):
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }

    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


def extract_text_pos(df):
    df.drop(['tweetId', 'userId', 'imageId(s)', 'username', 'timestamp'], axis=1, inplace=True)
    df['noun_count'] = df['tweetText'].apply(lambda x: check_pos_tag(x, 'noun'))
    df['verb_count'] = df['tweetText'].apply(lambda x: check_pos_tag(x, 'verb'))
    df['adj_count'] = df['tweetText'].apply(lambda x: check_pos_tag(x, 'adj'))
    df['adv_count'] = df['tweetText'].apply(lambda x: check_pos_tag(x, 'adv'))
    df['pron_count'] = df['tweetText'].apply(lambda x: check_pos_tag(x, 'pron'))
    df.drop(['tweetText'], axis=1, inplace=True)
    return df
