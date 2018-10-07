




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