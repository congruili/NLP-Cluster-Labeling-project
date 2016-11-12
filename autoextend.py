
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
    i = 0
    for d in json_data:
        sentence = d['tokens']
        for m in d['mentions']:
            entity = ' '.join(sentence[m['start'] : m['end']])
            for l in m['labels']:
                l = l.encode("utf-8")
                entity = entity.encode("utf-8")
                raw_classes[l].add(entity)
                raw_mentions[entity].add(l)
        	i += 1
        	if i >= 3000:
        		break
    	if i >= 3000:
        	break
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

train_embeddings = {}
for e in entities.keys():
    try:
        train_embeddings[e] = pre_embeddings[vocab[e]]
    except KeyError:
        pass
        continue

import tensorflow as tf
batch_size = 1
X_entities = tf.placeholder("float", [len(train_embeddings.keys()), 1])
X_recover = {}

def encoder(x):
    clusters = {}
    M = {}
    for k, ent in zip(range(len(train_embeddings.keys())), train_embeddings.keys()):
    	W = []
    	for i in range(len(entities[ent]) - 1):
    		w = tf.Variable(0.0)
    		M[(ent, entities[ent][i])] = tf.mul(X_entities[k], w)
    		W.append(w)
    	sum = tf.reduce_sum(W)
    	M[(ent, entities[ent][len(entities[ent]) - 1])] = tf.mul(X_entities[k], tf.sub(tf.constant(1.0), sum))

    for l in labels.keys():
        mem_list = [M[k] for k in M.keys() if k[1] == l]
       	if len(mem_list) > 0:
        	clusters[l] = tf.add_n(mem_list)
    print 'Encoding finished...'
    return clusters

def decoder(clusters):
    M = {}
    for k, l in zip(range(len(clusters.keys())), clusters.keys()):
    	W = []
    	for i in range(len(labels[l]) - 1):
    		w = tf.Variable(0.0)
    		M[(l, labels[l][i])] = tf.mul(clusters[l], w)
    		W.append(w)
    	sum = tf.reduce_sum(W)
    	M[(l, labels[l][len(labels[l]) - 1])] = tf.mul(clusters[l], tf.sub(tf.constant(1.0), sum))

    output = []
    for m in train_embeddings.keys():
        label_list = [M[k] for k in M.keys() if k[1] == m]
        output.append(tf.add_n(label_list))
    print 'Decoding finished...'
    return output

# Construct model
clusters = encoder(X_entities)
X_recover = decoder(clusters)

learning_rate = 0.01
# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.square(X_entities - X_recover))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

print 'Model is defined...'

import random

training_iters = 300000
default_batch_size = 1
display_step = 2
with tf.Session() as sess:
    if ans:
        sess.run(init)
        step = 0
        n_processed = 0
        n_processed_total = 0
        shuffled_idx = range(embedding_size)
        random.shuffle(shuffled_idx)
        while step < training_iters:
            if n_processed >= embedding_size:
                n_processed = 0
                shuffled_idx = range(embedding_size)
                random.shuffle(shuffled_idx)
            selected_idx = shuffled_idx[n_processed : min(n_processed + default_batch_size, embedding_size)][0]
            #n_processed += len(selected_idx)
            n_processed += 1
            #n_processed_total += len(selected_idx)
            n_processed_total += 1
            #batch_size = len(selected_idx)
            batch_size = 1
            fdict = {}
            X = []
            #for idx in selected_idx:
            X.append(np.array([train_embeddings[e][selected_idx] for e in train_embeddings.keys()]))
            #fdict[X_entities] = np.array(X).reshape((batch_size, len(train_embeddings.keys())))
            fdict[X_entities] = np.array(X).reshape((len(train_embeddings.keys()), 1))
            print step
            sess.run(optimizer, feed_dict=fdict)
            if step % display_step == 0:
                loss_val = sess.run(loss, feed_dict=fdict)
                print "Iter " + str(n_processed_total) + ", Minibatch Loss= " + "{:.6f}".format(loss_val)
            step += 1
        print "Optimization Finished!"
        save_path = saver.save(sess, "autoextent.ckpt")
        print("Model saved in file: %s" % save_path)
    else:
        saver.restore(sess, "autoextent.ckpt")
        print("Model restored.")