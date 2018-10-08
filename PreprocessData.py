# Dataset link: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from six.moves import cPickle as pickle

# nltk.download()


train = pd.read_csv("dataset/labeledTrainData.tsv", delimiter="\t")


# test = pd.read_csv("dataset/testData.tsv", delimiter="\t")


def clean_text(text, remove_stopwords=True):
    '''Clean the text, with the option to remove stopwords'''

    size = text.shape[0]
    reviews = np.empty(size, dtype=object)
    labels = np.zeros(size, dtype=int)

    index = 0
    for list in text:
        name = list[0]
        label = list[1]
        review = list[2].lower()
        reviews[index] = review
        labels[index] = label
        index = index + 1

    # Optionally, remove stop words
    if remove_stopwords:
        stopwords = np.loadtxt("dataset/stopwords.txt", dtype='str')

    # reviews = " ".join(reviews)
    for i in range(size):

        # Clean the text
        reviews[i] = re.sub(r"<br />", " ", reviews[i])
        reviews[i] = re.sub(r"[^a-z]", " ", reviews[i])
        reviews[i] = re.sub(r"   ", " ", reviews[i])  # Remove any extra spaces
        reviews[i] = re.sub(r"  ", " ", reviews[i])

        if remove_stopwords:
            word_list = reviews[i].split();
            reviews[i] = ' '.join([i for i in word_list if i not in stopwords])
            # for word in stopwords:
            #     reviews[i] = reviews[i].replace(" "+word+" ", " ")

    # Return a list of words
    return reviews, labels


train_clean, train_labels = clean_text(train.values, True)
# test_clean,test_labels=clean_text(test.values,True)


# Tokenize the reviews
# all_reviews = " ".join(train_clean)# + " ".join(test_clean)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_clean)
print("Fitting is complete.")

word_index = tokenizer.word_index
print("word index: " + str(word_index))

train_seq = tokenizer.texts_to_sequences(train_clean)
print("train_seq is complete.")

# test_seq = tokenizer.texts_to_sequences(test_clean)
# print("test_seq is complete")


# length covering 80% of the data
# length=np.percentile(train_seq, 80)

# print('length covering 80% : '+str(length))

max_review_length = 200

train_pad = pad_sequences(train_seq, maxlen=max_review_length)
print("train_pad is complete.")

# test_pad = pad_sequences(test_seq, maxlen = max_review_length)
# print("test_pad is complete.")


x_train, x_test, y_train, y_test = train_test_split(train_pad, train_labels, test_size=0.20, random_state=2)
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.20, random_state=2)

# save data to a pickle file to load when training

print("Saving data into pickle file")

pickle_file = 'dataset.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': x_train,
        'train_labels': y_train,
        'test_dataset': x_test,
        'test_labels': y_test,
        'valid_dataset': x_valid,
        'valid_labels': y_valid,
        'num_of_words': len(word_index),
        'tokenizer': tokenizer,
        'max_review_length':max_review_length
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Done")
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
