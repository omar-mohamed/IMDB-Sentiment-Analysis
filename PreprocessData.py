# Dataset link: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

import pandas as pd
import numpy as np
import tensorflow as tf
import nltk, re, time
from nltk.corpus import stopwords
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import namedtuple

# nltk.download()



train = pd.read_csv("dataset/labeledTrainData.tsv", delimiter="\t")
test = pd.read_csv("dataset/testData.tsv", delimiter="\t")


def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''

    size=text.shape[0]
    reviews = np.empty(size, dtype=object)
    labels=np.zeros(size, dtype=int)

    index=0
    for list in text:
        name=list[0]
        label=list[1]
        review=list[2].lower()
        reviews[index]=review
        labels[index]=label
        index=index+1


    # Convert words to lower case and split them
    reviews = np.array([x.lower() if isinstance(x, str) else x for x in reviews])

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        reviews = [w for w in reviews if not w in stops]

    # text = " ".join(text)

    # Clean the text
    reviews = re.sub(r"<br />", " ", reviews)
    reviews = re.sub(r"[^a-z]", " ", reviews)
    reviews = re.sub(r"   ", " ", reviews)  # Remove any extra spaces
    reviews = re.sub(r"  ", " ", reviews)

    # Return a list of words
    return reviews,labels


train_clean=clean_text(train.values,True)
test_clean=clean_text(test,True)


# Tokenize the reviews
all_reviews = train_clean + test_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

test_seq = tokenizer.texts_to_sequences(test_clean)
print("test_seq is complete")

word_index = tokenizer.word_index
print("word index:"+str(word_index))

#length covering 80% of the data
length=np.percentile(train_seq.counts, 80)

print('length covering 80% : '+str(length))

max_review_length = 200

train_pad = pad_sequences(train_seq, maxlen = max_review_length)
print("train_pad is complete.")

test_pad = pad_sequences(test_seq, maxlen = max_review_length)
print("test_pad is complete.")


