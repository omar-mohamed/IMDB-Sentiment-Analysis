from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorboard.backend.event_processing.event_accumulator import namedtuple
from tensorflow.contrib.layers.python.layers import initializers
from tqdm import tqdm


##################load data#####################

print("Loading data")

all_data = pickle.load(open('dataset.pickle', 'rb'))
train_data = all_data['train_dataset']
test_data = all_data['test_dataset']
valid_data = all_data['valid_dataset']

train_labels = all_data['train_labels']
test_labels = all_data['test_labels']
valid_labels = all_data['valid_labels']
n_words = all_data['num_of_words']

del all_data


def get_batches(x, y, batch_size):
    '''Create the batches for the training and validation data'''
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]


def build_rnn(n_words, embed_size, lstm_sizes, learning_rate, output_hidden_units):
    '''Build the Recurrent Neural Network'''

    tf.reset_default_graph()

    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        batch_size = tf.placeholder(tf.int32,shape=(), name='batch_size')
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.int32, [None, None], name='labels')


    # Create the embeddings
    with tf.name_scope("embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words + 1,
                                                   embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # Build the RNN layers
    with tf.name_scope("RNN_layers"):
        cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=n), output_keep_prob=keep_prob) for n
                 in lstm_sizes]

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # Set the initial state
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)
        # initial_state = tf.identity(initial_state, name="initial_state")

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(
            cell,
            embed,
            initial_state=initial_state,
        )

        # Create the fully connected layers
    with tf.name_scope("fully_connected"):
        # Initialize the weights and biases
        weights_initializer = initializers.xavier_initializer()
        biases = tf.zeros_initializer()

        dense = tf.contrib.layers.fully_connected(outputs[:, -1],
                                                  num_outputs=output_hidden_units[0],
                                                  activation_fn=tf.nn.relu,
                                                  weights_initializer=weights_initializer,
                                                  biases_initializer=biases)

        dense = tf.contrib.layers.dropout(dense, keep_prob)

        # Depending on the iteration, use a second fully connected layer
        for i in range(1, len(output_hidden_units)):
            dense = tf.contrib.layers.fully_connected(dense,
                                                      num_outputs=output_hidden_units[i],
                                                      activation_fn=tf.nn.relu,
                                                      weights_initializer=weights_initializer,
                                                      biases_initializer=biases)

            dense = tf.contrib.layers.dropout(dense, keep_prob)

    # Make the predictions
    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(dense,
                                                        num_outputs=1,
                                                        activation_fn=tf.sigmoid,
                                                        weights_initializer=weights_initializer,
                                                        biases_initializer=biases)

        predictions=tf.identity(predictions, name="predictions")

        tf.summary.histogram('predictions', predictions)

    # Calculate the cost
    with tf.name_scope('cost'):
        cost = tf.losses.sigmoid_cross_entropy(labels, predictions)
        tf.summary.scalar('cost', cost)

    # Train the model
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Determine the accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.cast(tf.round(predictions),
                                        tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'keep_prob','batch_size', 'initial_state',
                    'final_state', 'accuracy', 'predictions', 'cost',
                    'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def train(model, epochs, log_string):
    '''Train the RNN'''

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        valid_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./saved_model/logs/train/{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('./saved_model/logs/valid/{}'.format(log_string))

        for e in range(epochs):

            state = sess.run(model.initial_state,feed_dict={model.batch_size: batch_size})

            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []

            with tqdm(total=len(train_data)) as pbar:
                for _, (x, y) in enumerate(get_batches(train_data, train_labels, batch_size), 1):
                    feed = {model.inputs: x,
                            model.batch_size: batch_size,
                            model.labels: y[:, None],
                            model.keep_prob: dropout,
                            model.initial_state: state}
                    summary, loss, acc, state, _ = sess.run([model.merged,
                                                             model.cost,
                                                             model.accuracy,
                                                             model.final_state,
                                                             model.optimizer],
                                                            feed_dict=feed)

                    # Record the loss and accuracy of each training batch
                    train_loss.append(loss)
                    train_acc.append(acc)

                    # Record the progress of training
                    train_writer.add_summary(summary, iteration)

                    iteration += 1
                    pbar.update(batch_size)

            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc)

            val_state = sess.run(model.initial_state,feed_dict={model.batch_size: batch_size})
            with tqdm(total=len(valid_data)) as pbar:
                for x, y in get_batches(valid_data, valid_labels, batch_size):
                    feed = {model.inputs: x,
                            model.batch_size: batch_size,
                            model.labels: y[:, None],
                            model.keep_prob: 1,
                            model.initial_state: val_state}
                    summary, batch_loss, batch_acc, val_state = sess.run([model.merged,
                                                                          model.cost,
                                                                          model.accuracy,
                                                                          model.final_state],
                                                                         feed_dict=feed)

                    # Record the validation loss and accuracy of each epoch
                    val_loss.append(batch_loss)
                    val_acc.append(batch_acc)
                    pbar.update(batch_size)

            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)

            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            if avg_valid_loss > min(valid_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 2:
                    break

                    # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:
                print("New Record!")
                stop_early = 0
                checkpoint = "./saved_model/sentiment_{}.ckpt".format(
                    log_string)
                saver.save(sess, checkpoint)
        test_acc = []
        test_loss = []
        test_state = sess.run(model.initial_state,feed_dict={model.batch_size: batch_size})
        with tqdm(total=len(test_data)) as pbar:
            for x, y in get_batches(test_data, test_labels, batch_size):
                feed = {model.inputs: x,
                        model.batch_size: batch_size,
                        model.labels: y[:, None],
                        model.keep_prob: 1,
                        model.initial_state: test_state}
                summary, batch_loss, batch_acc, test_state = sess.run([model.merged,
                                                                      model.cost,
                                                                      model.accuracy,
                                                                      model.final_state],
                                                                      feed_dict=feed)

                # Record the validation loss and accuracy of each epoch
                test_loss.append(batch_loss)
                test_acc.append(batch_acc)
                pbar.update(batch_size)

            # Average the validation loss and accuracy of each epoch
        avg_test_loss = np.mean(test_loss)
        avg_test_acc = np.mean(test_acc)
        # Print the progress of each epoch
        print("Test Loss: {:.3f}".format(avg_test_loss),
              "Test Acc: {:.3f}".format(avg_test_acc))



# The default parameters of the model
embed_size = 300
batch_size = 100
lstm_sizes = [64]
dropout = 0.75
learning_rate = 0.0001
epochs = 50
output_hidden_units = [128]

log_string = 'lstm_sizes={},output_hidden_units={}'.format(lstm_sizes,
                                                           output_hidden_units)
model = build_rnn(n_words=n_words,
                  embed_size=embed_size,
                  lstm_sizes=lstm_sizes,
                  learning_rate=learning_rate,
                  output_hidden_units=output_hidden_units)

train(model, epochs, log_string)
