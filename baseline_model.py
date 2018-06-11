import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
from string import ascii_lowercase
import cv2
from random import shuffle
import math

FRAMES_PER_SIGN = 5

# Returns the placeholder objects for input data of shape [None, |n_H0|, |n_W0|, |n_C0|]
# and input labels of shape [None, |n_y|].
def get_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape = [None, n_y])

    return X, Y


# Returns weight parameters used for the neural network. Specifically, returns
# parameters W1 and W2.
def init_params():
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# Performs forward propagation on the model and returns the output of the last linear
# unit using the input dataset placeholder |X| and parameters |parameters|.
#
# Formally, implements the model below.
# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
def fwd_prop(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 26, activation_fn = None)

    return Z3


# Returns the cost produced by the model as found with last linear output |Z3|
# and true labels |Y|.
def calc_cost(Z3, Y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))


# Implements a three-layer CNN as defined by fwd_prop(). Uses |X_train| and
# |Y_train| as the training set and |X_test| and |Y_test| as the testing set with
# |learning_rate| learning rate. Runs |num_epochs| iterations with a minibatch size of
# |minibatch_size|. To print the cost per iteration, set |print_cost| to True.
# Returns the training accuracy, the testing accuracy, and the parameters learned.
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 10, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()
    m, n_H0, n_W0, n_C0 = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = get_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = init_params()
    Z3 = fwd_prop(X, parameters)
    cost = calc_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_minibatches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0: print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost and epoch % 1 == 0: costs.append(minibatch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


# Returns a list of random minibatches using |X| and |Y| as input data and labels, respectively.
# The minibatch sizes are set to be of size |mini_batch_size|.
def random_minibatches(X, Y, mini_batch_size = 64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# Converts label vectors into one-hot encodings.
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# Retrieves all frames from videos that are present in a local folder, randomly samples
# 5 frames per video, and finally returns a shuffled list of said frames.
def get_all_frames():
    files = [ c + str(num) for c in ascii_lowercase for num in range(2, 7) ]
    total_frames = []
    for f in files:
        name = f # |name| is not sufficient, here is where we would include the pathname to the database.
                 # Please note that we cannot link to the actual database due to a confidentiality agreement.

        vid_cap = cv2.VideoCapture(name)
        vid_frames = []

        success, frame = vid_cap.read()
        if frame is not None and frame.shape == (480, 640, 3): vid_frames.append(frame)

        while success:
            success, frame = vid_cap.read()
            if frame is not None and frame.shape == (480, 640, 3): vid_frames.append(frame)

        char = ord(f[0]) - ord('a')
        list_of_chars = [char] * FRAMES_PER_SIGN

        random_indices = list(np.random.choice(len(vid_frames), FRAMES_PER_SIGN, replace = False))
        chosen_frames = []
        for i in random_indices: chosen_frames.append(vid_frames[i])

        total_frames += zip(chosen_frames, list_of_chars)

    shuffle(total_frames)
    return total_frames


if __name__ == '__main__':
    all_frames = get_all_frames()
    num_frames = len(all_frames)
    num_training = int(num_frames * .7)
    num_testing = num_frames - num_training

    # Splitting up data into training and testing sets according to the split set above.
    lists = [[*x] for x in zip(*all_frames)]
    X, Y = lists[0], lists[1]
    X, Y = np.stack(np.array(X), axis = 0), np.stack(np.array(Y), axis = 0)

    X_train, X_test = X[:num_training] / 255., X[num_training:] / 255.
    Y_train, Y_test = convert_to_one_hot(Y[:num_training], 26).T, convert_to_one_hot(Y[num_training:], 26).T

    _, _, params = model(X_train, Y_train, X_test, Y_test)
