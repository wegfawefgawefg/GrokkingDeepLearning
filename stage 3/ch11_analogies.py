import numpy as np
from collections import Counter
import math
import random
import json

np.random.seed(1)
random.seed(1)

#   load layers
weights_0_1 = np.load("saved_layers/layer1.nn.npy")
weights_1_2 = np.load("saved_layers/layer2.nn.npy")

#   load word2index

#   if you dont save this then you arent looking at the same word indecies
#   #   its because "set" on line 18 doesnt preserve order.
#   #   so you have to save this then load it instead, inside any other case where you use it
'''
go recompute the word vectors, and make sure to save the word2index when you do it
'''
with open('word2index.json', 'r') as infile:
    word2index = json.load(infile)

#   why are we normalizing words vectors?
norms = np.sum(weights_0_1 * weights_0_1,axis=1)    
norms.resize(norms.shape[0],1)        
normed_weights = weights_0_1 * norms    

def w(vec):
    scores = Counter()
    for word,index in word2index.items():
        raw_difference = weights_0_1[index] - vec        
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)

def analogy(positive=['terrible','good'],negative=['bad']):        
    query_vect = np.zeros(len(weights_0_1[0]))    
    for word in positive:        
        query_vect += normed_weights[word2index[word]]    
    for word in negative:        
        query_vect -= normed_weights[word2index[word]]     
    return w(query_vect)

def similar(target='beautiful'):
    target_index = word2index[target]
    wordVec = weights_0_1[target_index]
    return w(wordVec)

def v(word):
    if word in word2index:
        return normed_weights[word2index[word]]
    else:
        return np.zeros(len(weights_0_1[0]))    

def powerLevel(strOrVec):
    if type(strOrVec) == type('str'):
        return np.linalg.norm(normed_weights[word2index[strOrVec]])
    else:
        return np.linalg.norm(strOrVec)


def rawAnal(positive=['terrible','good'],negative=['bad']):        
    norms = np.sum(weights_0_1 * weights_0_1,axis=1)    
    norms.resize(norms.shape[0],1)        
    normed_weights = weights_0_1 * norms        
    query_vect = np.zeros(len(weights_0_1[0]))    
    for word in positive:        
        query_vect += normed_weights[word2index[word]]    
    for word in negative:        
        query_vect -= normed_weights[word2index[word]]        
    scores = Counter()    
    for word,index in word2index.items():        
        raw_difference = weights_0_1[index] - query_vect        
        squared_difference = raw_difference * raw_difference        
        scores[word] = -math.sqrt(sum(squared_difference))            
    return scores.most_common(10)[1:]
