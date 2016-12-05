
msg = 'Do you want to retrain the model?'
ans = raw_input("%s (y/N) " % msg).lower() == 'y'

import os.path
from urllib import urlretrieve
import zipfile
def maybe_download(filename):
    """Download a file if not present"""
    if not os.path.exists(filename):
        url = 'http://nlp.stanford.edu/data/'
        filename, _ = urlretrieve(url + filename, filename)
        zip_ref = zipfile.ZipFile(os.path, 'r')
        zip_ref.extractall(os.path)
        zip_ref.close()
    return filename
#filename = maybe_download('glove.6B.zip')

import json
from collections import *
def get_data(json_data):
    classes = defaultdict(set)
    mentions = defaultdict(set)
    for d in json_data:
        sentence = d['tokens']
        for m in d['mentions']:
            entity = ' '.join(sentence[m['start'] : m['end']])
            for l in m['labels']:
                l = l.encode("utf-8")
                entity = entity.encode("utf-8")
                classes[l].add(entity)
                mentions[entity].add(l)
    return classes, mentions

import random
def get_training_and_test(classes, cut_percentage, percent_list):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    for key in classes:
        entities = list(classes[key])
        shuffled_idx = range(len(entities))
        random.shuffle(shuffled_idx)
        num_train = int(len(entities) * cut_percentage)
        basic_train_set = [entities[i] for i in shuffled_idx[:num_train]]
        basic_test_set = [entities[i] for i in shuffled_idx[num_train:]]
        for p in percent_list:
            shuffled_idx = range(len(basic_train_set))
            random.shuffle(shuffled_idx)
            num_pick = int(len(basic_train_set) * p * 0.01)
            train_data.append([basic_train_set[i] for i in shuffled_idx[:num_pick]])
            train_labels.append(label_dict[key])
        for p in percent_list:
            shuffled_idx = range(len(basic_test_set))
            random.shuffle(shuffled_idx)
            num_pick = int(len(basic_test_set) * p * 0.01)
            test_data.append([basic_test_set[i] for i in shuffled_idx[:num_pick]])
            test_labels.append(label_dict[key])
    return train_data, train_labels, test_data, test_labels

train_file = 'train.json'
with open(train_file) as f:
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    labels, entities = get_data(data)
label_dict = dict(zip(labels.keys(), range(len(labels.keys()))))
train_data, train_labels, test_data, test_labels = get_training_and_test(labels, 0.7, range(30, 100, 10))
print 'Size of training set: ', len(train_labels)
print 'Size of test set: ', len(test_labels)

import numpy as np
def get_pretrained_word_embeddings():
    vectors_file = 'glove.6B.100d.txt'
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(vectors)
    words = vectors.keys()
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v
    return vocab, W


num_classes = len(labels)
print 'Number of labels: ', num_classes

MAX_DOCUMENT_LENGTH = 150
vocab, pre_embeddings = get_pretrained_word_embeddings()
embedding_size = pre_embeddings.shape[1]

import string
def create_cluster_embeddings(entities, max_doc_length, pre_embeddings):
    embeddings = np.zeros((max_doc_length, embedding_size))
    idx = 0
    for word in entities:
        try:
            embeddings[idx, :] = pre_embeddings[vocab[word]]
        except KeyError:
            pass
            continue
        idx = idx + 1
        if idx == max_doc_length:
            break
    return embeddings

from sklearn.preprocessing import OneHotEncoder
def get_one_hot_labels(target_classes):
    enc = OneHotEncoder()
    return enc.fit_transform(np.array(target_classes).reshape(-1, 1)).toarray()
one_hot_labels = get_one_hot_labels(train_labels)
print 'One hot label shape: ', one_hot_labels.shape


#=========================================================================
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

X = []
y = []
for key in labels:
    entities = list(labels[key])
    X.append(pre_embeddings[vocab[e]] for e in entities)
    y.append(key for e in entities)

n_samples, n_features = X.shape
n_neighbors = 30

# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(i / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.show()
#=========================================================================
print hello


import tensorflow as tf
# Network Parameters
learning_rate = 0.001
training_iters = 100000
default_batch_size = 50
display_step = 100
n_input = embedding_size
n_steps = MAX_DOCUMENT_LENGTH
n_classes = one_hot_labels.shape[1]
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

EMBEDDING_SIZE = embedding_size
N_FILTERS = 100
N_FULL_CONNECT_OUTPUT = 100
WINDOW_SIZE = 3
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
POOLING_WINDOW = 4
POOLING_STRIDE = 1

# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)

# Create a convolution + maxpool layer for each filter size
filter_sizes = [2, 3, 4]
num_filters = 128

def conv_net(x):
    pooled_outputs = []
    x = tf.expand_dims(x, -1)
    for _, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            print filter_shape
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                padding="VALID", name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, MAX_DOCUMENT_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    return pooled_outputs

pooled_outputs = conv_net(x)
# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(3, pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
    pred_proc = tf.nn.softmax(scores)

# CalculateMean cross-entropy loss
l2_reg_lambda = 0.0
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores, y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    if ans:
        sess.run(init)
        step = 1
        n_processed = 0
        n_processed_total = 0
        shuffled_idx = range(len(train_labels))
        random.shuffle(shuffled_idx)
        
        while n_processed_total < training_iters:
            if n_processed >= len(train_labels):
                n_processed = 0
                shuffled_idx = range(len(train_labels))
                random.shuffle(shuffled_idx)
            selected_idx = shuffled_idx[n_processed:min(n_processed + default_batch_size, len(train_labels))]
            n_processed += len(selected_idx)
            n_processed_total += len(selected_idx)
            batch_size = len(selected_idx)
            batch_xs = np.array([create_cluster_embeddings(train_data[idx], MAX_DOCUMENT_LENGTH, pre_embeddings) for idx in selected_idx])
            batch_ys = np.array([one_hot_labels[idx] for idx in selected_idx])
            
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            fdict = {x: batch_xs, y: batch_ys, keep_prob: dropout}
            sess.run(optimizer, feed_dict=fdict)
            
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss_val = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(n_processed_total) + ", Minibatch Loss= " + "{:.6f}".format(loss_val) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        save_path = saver.save(sess, "cnn_cluster.ckpt")
        print("Model saved in file: %s" % save_path)
    else:
        # Restore variables from disk.
        saver.restore(sess, "cnn_cluster.ckpt")
        print("Model restored.")
    
    print 'Automatic Test Phase: '
    print 'Size: ', len(test_labels)
    batch_xs = np.array([create_cluster_embeddings(cluster, MAX_DOCUMENT_LENGTH, pre_embeddings) for cluster in test_data])
    batch_ys = np.array(one_hot_labels)
    batch_xs = batch_xs.reshape((len(test_labels), n_steps, n_input))
    fdict = {x: batch_xs, y: batch_ys, keep_prob: 1.}

    # Calculate test accuracy
    acc = sess.run(accuracy, feed_dict=fdict)
    print 'Test Accuracy= ' + '{:.5f}'.format(acc)
