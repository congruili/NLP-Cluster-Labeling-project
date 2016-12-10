
import json
from collections import *
import string

msg = 'Do you want to retrain the model?'
ans_retrain = raw_input("%s (y/N) " % msg).lower() == 'y'

msg = 'Do you want to continue training the model?'
ans_continue_train = raw_input("%s (y/N) " % msg).lower() == 'y'

msg = 'Do you want to load from file the training and test data?'
ans_resplit = raw_input("%s (y/N) " % msg).lower() == 'y'

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

def concat_terms(term):
    return term.replace(' ', '_')

from nltk.stem.porter import *
stemmer = PorterStemmer()
def check_hit(word, word_list):
    if word in word_list:
        return True
    # lowercase
    for candicate in word_list:
        if word.lower() == candicate.lower():
            return True
    # stemming
    for candicate in word_list:
        if stemmer.stem(word.lower()) == stemmer.stem(candicate.lower()):
            return True
    return False

def get_data(json_data):
    classes = defaultdict(set)
    mentions = defaultdict(set)
    all_classes = set()
    all_entities = set()
    for d in json_data:
        sentence = d['tokens']
        for m in d['mentions']:
            entity = ' '.join(sentence[m['start'] : m['end']])
            for l in m['labels']:
                l = concat_terms(l.encode("utf-8"))
                entity = concat_terms(entity.encode("utf-8"))
                classes[l].add(entity)
                mentions[entity].add(l)
                all_classes.add(l)
                all_entities.add(entity)
    return classes, mentions, all_classes, all_entities

from embedding import PreTrainEmbedding
embedding_size = 300
embedding = PreTrainEmbedding('/Users/jason/EclipseWorkspace/GoogleNews-vectors-negative300.bin.gz', embedding_size)

def check_embedding_coverage(labels, entities):
    count_label = 0
    hit = 0
    for l in labels:
        for t in l.split(':'):
            count_label += 1
            if embedding.get_embedding(t) is not None:
                hit += 1
    print 'Coverage for labels: ', float(hit)/count_label

    hit = 0
    for ent in entities:
        if embedding.get_embedding(ent) is not None:
            hit += 1
    print 'Coverage for entities: ', float(hit)/len(entities.keys())

import cPickle as pickle
def save_data(data, file_name):
    with open(file_name, 'wb') as outfile:
        pickle.dump(data, outfile)

def recover_data(file_name):
    data = pickle.load(open(file_name, 'rb'))
    return data

import random
def get_training_and_test(from_archive, classes, cut_percentage, percent_list):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    if from_archive:
        train_data = recover_data('train_data.p')
        test_data = recover_data('test_data.p')
        train_labels = recover_data('train_labels.p')
        test_labels = recover_data('test_labels.p')
        return train_data, train_labels, test_data, test_labels

    for key in classes:
        k = key.split(':')
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
            train_labels.append(k[0])
            if len(k) > 1:
                train_data.append([basic_train_set[i] for i in shuffled_idx[:num_pick]])
                train_labels.append(k[1])
        for p in percent_list:
            shuffled_idx = range(len(basic_test_set))
            random.shuffle(shuffled_idx)
            num_pick = int(len(basic_test_set) * p * 0.01)
            test_data.append([basic_test_set[i] for i in shuffled_idx[:num_pick]])
            test_labels.append(k[0])
            if len(k) > 1:
                test_data.append([basic_test_set[i] for i in shuffled_idx[:num_pick]])
                test_labels.append(k[1])
    save_data(train_data, 'train_data.p')
    save_data(test_data, 'test_data.p')
    save_data(train_labels, 'train_labels.p')
    save_data(test_labels, 'test_labels.p')
    return train_data, train_labels, test_data, test_labels

train_file = 'train.json'
with open(train_file) as f:
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    labels, entities, all_classes, all_entities = get_data(data)
    check_embedding_coverage(labels, entities)
    train_data, train_labels, test_data, test_labels = get_training_and_test(ans_resplit, labels, 0.7, range(30, 100, 10))
print 'Size of training set: ', len(train_labels)
print 'Size of test set: ', len(test_labels)
print 'Number of labels: ', len(labels)
print 'Number of clusters: ', len(all_classes)
print 'Number of entities: ', len(all_entities)

def create_cluster_embeddings(entities, max_doc_length):
    embeddings = np.zeros((max_doc_length, embedding_size))
    idx = 0
    for word in entities:
        if ':' in word:
            word = word.split(':')[1].lower()
        if len(word.split()) > 1:
            word = word.replace(' ', '_').lower()
        result = embedding.get_embedding(word)
        if result is not None:
            embeddings[idx, :] = result
        idx = idx + 1
        if idx == max_doc_length:
            break
    return embeddings

def create_label_embedding(label):
    embed = np.zeros(embedding_size)
    if ':' in label:
        label = label.split(':')[1].lower()
    if len(label.split()) > 1:
        label = label.replace(' ', '_').lower()
    result = embedding.get_embedding(label)
    if result is not None:
        embed = result
    return embed

import numpy as np
import tensorflow as tf
# Network Parameters
learning_rate = 0.01
training_iters = 100000
default_batch_size = 50
display_step = 5000
n_input = embedding_size
MAX_DOCUMENT_LENGTH = 5
n_steps = MAX_DOCUMENT_LENGTH
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_input])
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
        shape=[num_filters_total, embedding_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    pred_y = tf.nn.xw_plus_b(h_drop, W, b, name="pred_y")
    predictions = tf.argmax(pred_y, 1, name="predictions")
    pred_proc = tf.nn.softmax(pred_y)

# CalculateMean cross-entropy loss
l2_reg_lambda = 0.3
with tf.name_scope("loss"):
    losses = tf.reduce_mean(tf.square(pred_y - y))
    loss = losses + l2_reg_lambda * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    if ans_retrain:
        if ans_continue_train:
            saver.restore(sess, "cnn_embedding.ckpt")
            print("Model restored.")
        else:
            sess.run(init)
        step = 1
        n_processed = 0
        n_processed_total = 0
        shuffled_idx = range(len(train_labels))
        random.shuffle(shuffled_idx)
        
        while step < 50000:
            if n_processed >= len(train_labels):
                n_processed = 0
                shuffled_idx = range(len(train_labels))
                random.shuffle(shuffled_idx)
            selected_idx = shuffled_idx[n_processed:min(n_processed + default_batch_size, len(train_labels))]
            n_processed += len(selected_idx)
            n_processed_total += len(selected_idx)
            batch_size = len(selected_idx)
            batch_xs = np.array([create_cluster_embeddings(train_data[idx], MAX_DOCUMENT_LENGTH) for idx in selected_idx])
            batch_ys = np.array([create_label_embedding(train_labels[idx]) for idx in selected_idx])
            
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            fdict = {x: batch_xs, y: batch_ys, keep_prob: dropout}
            sess.run(optimizer, feed_dict=fdict)
            
            if step % display_step == 0:
                hit = 0
                y_pred = sess.run(pred_y, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                for i in range(len(y_pred)):
                    #words = embedding.get_similar_words_from_candidates(y_pred[i])
                    words = embedding.get_similar_words(y_pred[i])
                    #print words
                    #print train_labels[selected_idx[i]]
                    if check_hit(train_labels[selected_idx[i]], words):
                        hit += 1
                accuracy = float(hit)/batch_size

                # Calculate batch loss
                loss_val = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss_val) + \
                      ", Training Accuracy= " + "{:.5f}".format(accuracy)

            if step % 50000 == 0:
                msg = 'Do you want to stop training the model?'
                if raw_input("%s (y/N) " % msg).lower() == 'y':
                    break
            step += 1
        print "Optimization Finished!"
        save_path = saver.save(sess, "cnn_embedding.ckpt")
        print("Model saved in file: %s" % save_path)
    else:
        # Restore variables from disk.
        saver.restore(sess, "cnn_embedding.ckpt")
        print("Model restored.")
    
    print 'Automatic Test Phase: '
    print 'Size: ', len(test_labels)
    batch_xs = np.array([create_cluster_embeddings(cluster, MAX_DOCUMENT_LENGTH) for cluster in test_data])
    batch_ys = np.array([create_label_embedding(l) for l in test_labels])
    batch_xs = batch_xs.reshape((len(test_labels), n_steps, n_input))
    fdict = {x: batch_xs, y: batch_ys, keep_prob: 1.}

    hit = 0
    y_pred = sess.run(pred_y, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
    
    with open('extend_label.json') as f:
        basic_truth = json.load(f)

    truth = defaultdict(list)
    results = defaultdict(list)
    for i in range(len(y_pred)):
        #words = embedding.get_similar_words_from_candidates(y_pred[i], topn=2)
        words = embedding.get_similar_words(y_pred[i], topn=10)
        print 'Cluster Entities: ', test_data[i]
        print 'True Label: ', test_labels[i]
        print 'Suggested Labels: ', words
        print ''
        if check_hit(test_labels[i], words):
            hit += 1
        results[i].extend(words)

        if test_labels[i] in basic_truth.keys():
            for s in basic_truth[test_labels[i]]:
                truth[i].append(s)
        truth[i].append(test_labels[i])

    accuracy = float(hit)/len(test_labels)
    print 'Test Accuracy= ' + '{:.5f}'.format(accuracy)

    from evals import *
    match_k = match_at_K(truth, results, 1)
    print 'match_at_k = ' + '{:.5f}'.format(match_k)

    mrr_k = mrr_at_K(truth, results, 1)
    print 'mrr_at_k = ' + '{:.5f}'.format(mrr_k)


