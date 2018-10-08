
import numpy as np
import re
from six.moves import cPickle as pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

str_input="this is a very bad movie!"

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
max_review_length=all_data['max_review_length']
del all_data

str_input=clean_text(str_input)


input_seq = tokenizer.texts_to_sequences(str_input)

input_pad = pad_sequences(input_seq, maxlen=max_review_length)

model_saver = tf.train.import_meta_graph('./best_model/sentiment_lstm_sizes=[64],output_hidden_units=[128].ckpt.meta')

def classifyReview( input):
    with tf.Session() as sess:
        # new_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')
        model_saver.restore(sess, tf.train.latest_checkpoint('./best_model/'))
        graph = sess.graph
        inputs = graph.get_tensor_by_name("inputs/inputs:0")
        batch_size = graph.get_tensor_by_name("inputs/batch_size:0")
        initial_state = graph.get_tensor_by_name("RNN_init_state/initial_state:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        predictions= graph.get_tensor_by_name("predictions/predictions:0")
        np_input=np.zeros(1,max_review_length)
        np_input[0]=input

        zero_state = sess.run(
            [initial_state], feed_dict={batch_size:1})
        # print([node.name for node in graph.as_graph_def().node])

        feed_dict = {inputs: np_input,keep_prob: 1,batch_size:1,initial_state:zero_state}
        predictions = sess.run(
            [predictions], feed_dict=feed_dict)
        return predictions[0]


answer=classifyReview(input_pad)






























# def make_predictions(lstm_size, multiple_fc, fc_units, checkpoint):
#     '''Predict the sentiment of the testing data'''
#
#     # Record all of the predictions
#     all_preds = []
#
#     model = build_rnn(n_words=n_words,
#                       embed_size=embed_size,
#                       batch_size=batch_size,
#                       lstm_size=lstm_size,
#                       num_layers=num_layers,
#                       dropout=dropout,
#                       learning_rate=learning_rate,
#                       multiple_fc=multiple_fc,
#                       fc_units=fc_units)
#
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         # Load the model
#         saver.restore(sess, checkpoint)
#         test_state = sess.run(model.initial_state)
#         for _, x in enumerate(get_test_batches(x_test, batch_size), 1):
#             feed = {model.inputs: x,
#                     model.keep_prob: 1,
#                     model.initial_state: test_state}
#             predictions = sess.run(model.predictions, feed_dict=feed)
#             for pred in predictions:
#                 all_preds.append(float(pred))
#
#
#     return all_preds


# checkpoint1 = "./sentiment_ru=128,fcl=False,fcu=256.ckpt"
#
#
# predictions1 = make_predictions(128, False, 256, checkpoint1)