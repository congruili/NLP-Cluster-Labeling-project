import numpy as np
import gensim

class PreTrainEmbedding():
    def __init__(self, file, embedding_size):
        self.embedding_size = embedding_size
        self.model = gensim.models.Word2Vec.load_word2vec_format(file, binary=True)

    def get_embedding(self, word):
        try:
            result = self.model[word]
            return result
        except KeyError:
            print 'Can not get embedding for ', word
            return None

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

        