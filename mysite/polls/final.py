import nltk, re, string
from nltk.corpus import stopwords
import numpy as np
import pickle

with open('polls/freqs.pickle', 'rb') as handle:
    freqs = pickle.load(handle)

with open('polls/theta.pickle', 'rb') as handle:
    theta = pickle.load(handle)

def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


#  Logistic regression

# Sigmoid Function
def sigmoid(z):
    """
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    """
    zz = np.negative(z)
    h = 1 / (1 + np.exp(zz))
    return h


#  Extracting the features

def extract_features(tweet, freqs):
    """
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    """

    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert (x.shape == (1, 3))
    return x


# test on training data
"""
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)
"""


def predict_tweet(tweet, freqs, theta):
    """
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    """
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))

    return y_pred


# Predict with your own tweet
def pre(sentence):
    yhat = predict_tweet(sentence, freqs, theta)
    if yhat > 0.5:
        return 'Positive sentiment'
    elif yhat == 0:
        return 'Neutral sentiment'
    else:
        return 'Negative sentiment'