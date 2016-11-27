import sys
import networkx as nx
import operator

def buildGraph(finished_words):
    G = nx.DiGraph()
    for word in finished_words:
        results = finished_words[word]
        components = set()
        for result in results:
            addElement(components, result, 'typeOf')
            addElement(components, result, 'instanceOf')
            addElement(components, result, 'memberOf')
            addElement(components, result, 'partOf')
            addElement(components, result, 'regionOf')
            addElement(components, result, 'usageOf')
            addElement(components, result, 'substanceOf')
            addElement(components, result, 'inCategory')
            addElement(components, result, 'synonyms')
            addElement(components, result, 'entails')
            addElement(components, result, 'also')
            addElement(components, result, 'similarTo')
            addElement(components, result, 'pertainsTo')
        for component in components:
            G.add_edge(word, component)
    return G

def addElement(components, result, keyword):
    if keyword not in result: return
    elements = result[keyword]
    for e in elements:
        components.add(e)

def loadData(dirPath):
    G = nx.read_gpickle(dirPath+"/graph.dat")
    
    with open(dirPath+"/target.dat") as f:
        content = f.read()
        target = eval(content)
    
    with open(dirPath+"/finished_words.dat") as f:
        content = f.read()
        finished_words = eval(content)
    print len(finished_words.keys()), len(G.nodes())

    return G, target, finished_words

def bfsSearch(G, target):
    max_length = 4
    res = {node:{target_word:max_length for target_word in target} for node in G.nodes()}
    for target_word in target:
        preVisitedNodes = set()
        curVisitedNodes = set([target_word])
        postVisitedNodes = set()
        path_length = 1

        while len(curVisitedNodes)>0 and path_length<max_length:
            print len(preVisitedNodes), len(curVisitedNodes), len(postVisitedNodes)
            for word in curVisitedNodes:
                if word in preVisitedNodes: continue
                if word not in G.nodes(): continue
                for next_word in G.neighbors(word):
                    if next_word in preVisitedNodes or next_word in curVisitedNodes: continue
                    postVisitedNodes.add(next_word)
                    res[next_word][target_word]=path_length
            path_length+=1
            preVisitedNodes.update(curVisitedNodes)
            curVisitedNodes = postVisitedNodes
            postVisitedNodes = set()

        sorted_res = sorted(res.items(), key=lambda x: sum(x[1].values()))
        print [(key,sum(value.values())) for (key, value) in sorted_res[:10]]
    #print [(key,sum(value.values())) for (key, value) in sorted_res[:10]]
    #print sorted_res[:10]

dirPath=sys.argv[1]
G, target, finished_words = loadData(dirPath)
#G = buildGraph(finished_words)
#print len(G.nodes())
bfsSearch(G, target)

