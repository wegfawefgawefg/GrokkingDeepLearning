import numpy as np
from collections import Counter
import sys
import random
import math
import re
import string 
from pprint import pprint

#########################################################
path = "F:\\_SSD_CODEING\\GrokkingDeepLearning\\stage 3\\babi\\tasksv11\\en\\qa1_single-supporting-fact_train.txt"
with open(path, 'r') as f:
    raw = f.readlines()

lines = []
for line in raw[:1000]:
    cleaned = line.lower().replace("\n", "").replace("\t", "")
    chars = [" " + char if not char.isalpha() else char for char in cleaned]
    chars = ''.join(chars)
    tokens = chars.split(' ')
    tokens = [token for token in tokens if not len(token) == 0]
    tokens = [token for token in tokens if token not in string.punctuation ]
    tokens = tokens[1:]
    if tokens[-1].isnumeric():
        tokens = tokens[:-1]
    lines.append( tokens )

print("Data sample: ")
pprint(lines[:3])
print("# " * 10)

#########################################################
def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex/ex.sum(axis=0)

vocab = set()
for line in lines:
    for word in line:
        vocab.add(word)
    
wordToIndex = {}
for i, word in enumerate(vocab):
    wordToIndex[word] = i

def wordsToIndices(words):
    indices = [wordToIndex[word] for word in words]
    return indices

#########################################################
np.random.seed(1)
embeddingSize = 10

embeddings = 0.1 * (np.random.rand(len(vocab), embeddingSize) - 0.5)

recurrent = np.eye(embeddingSize)

start = np.zeros(embeddingSize)

decoder =  0.1 * (np.random.rand(embeddingSize, len(vocab)) - 0.5)

oneHot = np.eye(len(vocab))

#########################################################
