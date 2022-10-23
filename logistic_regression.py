import nltk
import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
from nltk.corpus import twitter_samples 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



nltk.download('twitter_samples')
nltk.download('stopwords')

def process_tweet(tweet):

    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')

    tweet = re.sub(r'\$\w*','',tweet)
    tweet = re.sub(r'^RT[/s]+','',tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
    tweet = re.sub(r'#','',tweet)

    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []

    for word in tweet_tokens:
        if (word not in stopwords_en and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    
    return tweets_clean


def build_freqs(tweets,ys):

    yslist = np.squeeze(ys).tolist()
    freqs = {}

    for y,tweet in zip(yslist,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)

            if pair in freqs:
                freqs[pair]+=1
            else:
                freqs[pair]=1
    
    return freqs



all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos+train_neg
test_x = test_pos+test_neg

train_y = np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),1)),axis=0)
test_y = np.append(np.ones((len(test_pos),1)),np.zeros((len(test_neg),1)),axis=0)

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))


freqs = build_freqs(train_x,train_y)

print('Tweet: \n', train_x[0])
print('\nProcessed tweet: \n', process_tweet(train_x[0]))

def extract_features(tweet,freqs):

    tweet = process_tweet(tweet)

    x=np.zeros((1,3))

    x[0,0] = 1

    for word in tweet:
        x[0,1]+=freqs.get((word,1.0),0)
        x[0,2]+=freqs.get((word,0.0),0)
    
    assert(x.shape==(1,3))

    return x

X = np.zeros((len(train_x),3))
for i in range((len(train_x))):
    X[i,:] = extract_features(train_x[i],freqs)


Y=train_y

def eval(y_test,y_pred):
    print("Accuracy score:",accuracy_score(y_test,y_pred))
    print("Precision score:",precision_score(y_test,y_pred))
    print("Recall Score:",recall_score(y_test,y_pred))
    print("Confusion Matrix:",confusion_matrix(y_test,y_pred))

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X,Y.ravel())

def predict(x):
    X = np.zeros((len(x),3))
    for i in range((len(x))):
        X[i,:] = extract_features(x[i],freqs)
    classifier_pred = classifier.predict(X)
    return classifier_pred


predictions = predict(test_x)

eval(test_y,predictions)

p = predict(["This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!"])
