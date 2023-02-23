'''

import nltk, re, string
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pickle


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



def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


# create frequency dictionary
freqs = build_freqs(train_x, train_y)

with open('freqs.pickle', 'wb') as handle:
    pickle.dump(freqs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# check the output
"""
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

"""
# test the function below
"""
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))
"""

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

# Cost function and Gradient
def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    """
    # get 'm', the number of rows in matrix x
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        # calculate the cost function
        cost = -1. / m * (np.dot(y.transpose(), np.log(h)) + np.dot((1 - y).transpose(), np.log(1 - h)))
        # update the weights theta
        theta = theta - (alpha / m) * np.dot(x.transpose(), (h - y))

    cost = float(cost)
    return cost, theta


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

# Training Your Model

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
# these values are predefined (Andrew NG)
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

with open('theta.pickle', 'wb') as handle:
    pickle.dump(theta, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)

    return accuracy


tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
# print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

# Predict with your own tweet

my_tweet = 'It is so hot today but it is the perfect day for a beach party'
# print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)

def pre(sentence):
    yhat = predict_tweet(sentence, freqs, theta)
    if yhat > 0.5:
        return 'Positive sentiment'
    elif yhat == 0:
        return 'Neutral sentiment'
    else:
        return 'Negative sentiment'

'''