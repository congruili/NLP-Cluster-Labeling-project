
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
    raw_classes = defaultdict(set)
    raw_mentions = defaultdict(set)
    for d in json_data:
        sentence = d['tokens']
        for m in d['mentions']:
            entity = ' '.join(sentence[m['start'] : m['end']])
            for l in m['labels']:
                l = l.encode("utf-8")
                entity = entity.encode("utf-8")
                raw_classes[l].add(entity)
                raw_mentions[entity].add(l)
    classes = defaultdict(list)
    mentions = defaultdict(list)
    for l in raw_classes.keys():
        classes[l] = list(raw_classes[l])
    for m in raw_mentions.keys():
        mentions[m] = list(raw_mentions[m])
    return classes, mentions

train_file = 'train.json'
with open(train_file) as f:
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    labels, entities = get_data(data)
    print 'clusters: ', len(labels)
    print 'mentions: ', len(entities)
label_dict = dict(zip(labels.keys(), range(len(labels.keys()))))

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
vocab, pre_embeddings = get_pretrained_word_embeddings()
embedding_size = pre_embeddings.shape[1]

num_of_clusters = len(labels)
num_of_entities = len(entities)
print 'Number of labels: ', num_of_clusters
print 'embedding size: ', embedding_size

train_embeddings = {}
for e in entities.keys():
    try:
        train_embeddings[e] = pre_embeddings[vocab[e]]
    except KeyError:
        pass
        continue
print 'Actual entity size: ', len(train_embeddings.keys())

import tensorflow as tf
X_entities = tf.placeholder("float", [len(train_embeddings.keys()), embedding_size])
X_recover = {}

import collections
def encoder(x):
    clusters = {}
    M = collections.defaultdict(list)
    for k, ent in zip(range(len(train_embeddings.keys())), train_embeddings.keys()):
        if len(entities[ent]) == 1:
            M[entities[ent][0]].append(X_entities[k, :])
        else:
            W = []
            for i in range(len(entities[ent]) - 1):
                w = tf.Variable(tf.random_normal([embedding_size]), dtype=tf.float32)
                M[entities[ent][i]].append(tf.mul(X_entities[k, :], w))
                W.append(w)
            sum = tf.add_n(W)
            M[entities[ent][len(entities[ent]) - 1]].append(tf.mul(X_entities[k, :], tf.sub(tf.ones([embedding_size], dtype=tf.float32), sum)))

    for l in labels.keys():
        if len(M[l]) > 0:
            clusters[l] = tf.add_n(M[l])
    print 'Encoding finished...'
    return clusters

def decoder(clusters):
    M = collections.defaultdict(list)
    for k, ent in zip(range(len(train_embeddings.keys())), train_embeddings.keys()):
        if len(entities[ent]) == 1:
            M[ent].append(clusters[entities[ent][0]])
        else:
            W = []
            for i in range(len(entities[ent]) - 1):
                w = tf.Variable(tf.random_normal([embedding_size]), dtype=tf.float32)
                M[ent].append(tf.mul(clusters[entities[ent][i]], w))
                W.append(w)
            sum = tf.add_n(W)
            M[ent].append(tf.mul(clusters[entities[ent][len(entities[ent]) - 1]], tf.sub(tf.ones([embedding_size], dtype=tf.float32), sum)))

    output = []
    for ent in train_embeddings.keys():
        output.append(tf.add_n(M[ent]))
    print 'Decoding finished...'
    return output

# Construct model
clusters = encoder(X_entities)
X_recover = decoder(clusters)

learning_rate = 0.01
loss = tf.reduce_mean(tf.square(X_entities - X_recover))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.initialize_all_variables()
saver = tf.train.Saver()

print 'Model is defined...'

import random
training_iters = 300000
display_step = 50
with tf.Session() as sess:
    if ans:
        sess.run(init)
        step = 0
        while step < training_iters:
            fdict = {}
            X = np.array([train_embeddings[e] for e in train_embeddings.keys()])
            fdict[X_entities] = X.reshape((len(train_embeddings.keys()), embedding_size))
            sess.run(optimizer, feed_dict=fdict)
            if step % display_step == 0:
                loss_val = sess.run(loss, feed_dict=fdict)
                print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss_val)
            step += 1
        print "Optimization Finished!"
        save_path = saver.save(sess, "autoextend.ckpt")
        print("Model saved in file: %s" % save_path)
    else:
        saver.restore(sess, "autoextend.ckpt")
        print("Model restored.")
