import json
import gensim
import numpy
from collections import *
import pickle

model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
final_context = ""
missing = 0
total = 0
new_training_data = []

def get_data(json_data):
    global model, final_context, missing, total, new_training_data
    classes = defaultdict(set)
    mentions = defaultdict(set)
    for d in json_data:
        sentence = d['tokens']
        # count_words += len(sentence)
        new_training_data.append(sentence)
        context = ' '.join(sentence)
        final_context += context
        final_context += " "

        for m in d['mentions']:
            entity = '_'.join(sentence[m['start'] : m['end']])
            total += 1
            entity = entity.encode("utf-8")
            if not model.vocab.has_key(entity):
                missing += 1

            for l in m['labels']:
                l = l.encode("utf-8")
                classes[l].add(entity)
                mentions[entity].add(l)
    return classes, mentions

def main():  
    global model, final_context, missing, total, new_training_data
    train_file = 'train.json' 

    with open(train_file) as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))
        labels, entities = get_data(data)
    # label_dict = dict(zip(labels.keys(), range(len(labels.keys()))))
    # key = labels.keys()[2]
    # print(key)
    # print(labels[key])
    outfile = open("context.txt", 'w+')
    outfile.write(final_context)
    outfile.close()
    print('Missing entities in Google News pretrained vectors: {:.2%}'.format(float(missing) / total))
    pickle.dump( labels, open( "labels.p", "wb+" ) )
    pickle.dump( entities, open( "entities.p", "wb+" ) )

    # print("starting training")
    # model.syn0lockf = numpy.ones(len(model.syn0), dtype=numpy.float32) 
    # model.train(new_training_data, total_words = count_words, total_examples = len(new_training_data))
    # model.save('my_model')    
    
if __name__ == "__main__":main()
