import numpy as np
import gensim
import cPickle as pickle
from collections import *
from scipy import spatial
import string

class PreTrainEmbedding():
    def __init__(self, file, embedding_size):
        self.embedding_size = embedding_size
        self.model = gensim.models.Word2Vec.load_word2vec_format(file, binary=True)

        hypernym_pairs = pickle.load(open('label_cands.p', 'rb'))
        print hypernym_pairs
        self.candidates = set()
        for p in hypernym_pairs:
            for w in hypernym_pairs[p]:
                self.candidates.add(w)
        self.candidates  = list(self.candidates)
        # self.candidates = set()
        # for p in hypernym_pairs:
        #     self.candidates.add(p[0])
        #     self.candidates.add(p[1])
        # self.candidates  = list(self.candidates)
        print self.candidates
        print len(self.candidates)

    def get_embedding(self, word):
        word_list = [word, word.upper(), word.lower(), string.capwords(word, '_')]

        tokens = word.split('_')
        if len(tokens) > 1:
            word_list.append(tokens[-1].upper())
            word_list.append(tokens[-1].lower())

        for w in word_list:
            try:
                result = self.model[w]
                return result
            except KeyError:
                #print 'Can not get embedding for ', w
                continue
        return None

    def get_similar_words(self, embedding, topn=3):
        words = self.model.similar_by_vector(embedding, topn=topn)
        result = [w[0] for w in words]
        return result

    def get_similar_words_from_candidates(self, embedding, topn=3):
        scores = []
        for c in self.candidates:
            emb_cand = self.get_embedding(c)
            # Compute cosine similarity
            s = 1 - spatial.distance.cosine(emb_cand, embedding)
            scores.append(s)
        indices = np.argsort(np.array(scores))[::-1][:topn]
        result = [self.candidates[i] for i in indices]
        return result

    def get_glove_embedding():
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

        