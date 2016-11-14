# Unsupervised approaches using NLTK WordNet interface

Documentation of NLTK WordNet interface is available [here](http://www.nltk.org/howto/wordnet.html). </br>
Google news pre-trained vectors are available [here] (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing).

`pre_process.py` does a few things on the training data `train.json`:
* calculate the percentage of missing entities in Google News pretrained vectors: **10.27%**.
* glue the context of the training data together and save it as a new file `context.txt`.
* save the {label: [entities]} dict in a file `labels.p` and the {entity: [labels]} dict in another file `entities.p`. (later on the dicts could be loaded directly from these two files.)

Each entity in a labeled cluster has multiple synsets. Each synset represents a specific sense of this entity.
For example, entity a has synsets a1, a2, and a3; entity b has synsets b1 and b2 … 

Three approaches so far:
* `hypernyms.py` obtains statistics of the hypernyms of all the synsets. (Hypernyms of a1, a2, a3, b1, b2 are all obtained if they exist.)

* `lch.py` obtains statistics of the lowest common hypernyms for every pairwise combination of synsets between each pair of entities in the cluster. (LCHs of (a1 & b1) (a1 & b2) (a2 & b1) (a2 & b2) (a3 & b1) (a3 & b2) are all obtained if they exist.)

* `bfs.py` gets the full paths from the synset  to the root for each synset (each synset could have multiple paths to the root), then obtains the statistics for the first k layers starting from the synsets going up (k is at most 10), breaks when the frequency of the most common hypernym so far is larger than 80% of the number of entities in the cluster (which means more than 80% of entities in the cluster share the same hypernym). 
