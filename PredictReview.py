import numpy as np
import re
from six.moves import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

str_input = "this is a very bad movie!"


def clean_text(text, remove_stopwords=True):
    # Optionally, remove stop words
    if remove_stopwords:
        stopwords = np.loadtxt("dataset/stopwords.txt", dtype='str')

    # Clean the text
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"   ", " ", text)  # Remove any extra spaces
    text = re.sub(r"  ", " ", text)

    if remove_stopwords:
        word_list = text.split();
        text = ' '.join([i for i in word_list if i not in stopwords])

    return text


all_data = pickle.load(open('dataset.pickle', 'rb'))
tokenizer = all_data['tokenizer']
max_review_length = all_data['max_review_length']
del all_data

str_input = clean_text(str_input)

input_seq = tokenizer.texts_to_sequences([str_input])

input_pad = pad_sequences(input_seq, maxlen=max_review_length)

model_saver = tf.train.import_meta_graph("./best_model/sentiment_lstm_sizes=[64],output_hidden_units=[128].ckpt.meta")


def classifyReview(input):
    with tf.Session() as sess:
        # new_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')
        model_saver.restore(sess, tf.train.latest_checkpoint('./best_model/'))
        graph = sess.graph
        inputs = graph.get_tensor_by_name("inputs/inputs:0")
        batch_size = graph.get_tensor_by_name("inputs/batch_size:0")

        keep_prob = graph.get_tensor_by_name("inputs/keep_prob:0")
        predictions = graph.get_tensor_by_name("predictions/predictions:0")

        np_input = np.zeros((1, max_review_length), dtype=int)
        np_input[0] = input

        feed_dict = {inputs: np_input, keep_prob: 1, batch_size: 1}

        predictions = sess.run(
            [predictions], feed_dict=feed_dict)
        return predictions[0]


answer = classifyReview(input_pad)

if answer >= 0.5:
    print("Good movie")
else:
    print("Bad movie")

