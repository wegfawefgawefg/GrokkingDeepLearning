from collections import Counter
import math
import numpy as np
import json
from pprint import pprint

path2 = "F:\_SSD_CODEING\GrokkingDeepLearning\stage 3\wordToIndex.json"
with open(path2, 'r') as inFile:
    wordToIndex = json.load(inFile)

layer1 = np.load("saved_layers/layer1.nn.npy")

def fetchSimilar(target = 'beautiful'):
    targetIndex = wordToIndex[target]
    scores = Counter()
    for word, index in wordToIndex.items():
        delta = layer1[index] - layer1[targetIndex]
        deltaSquared = delta * delta
        scores[word] = -math.sqrt(sum(deltaSquared))
    return scores.most_common(10)

def fetchDifferent(target = 'beautiful'):
    targetIndex = wordToIndex[target]
    scores = Counter()
    for word, index in wordToIndex.items():
        delta = layer1[index] - layer1[targetIndex]
        deltaSquared = delta * delta
        scores[word] = -math.sqrt(sum(deltaSquared))
    return list(reversed(list(scores.most_common())))[:10]

# target = 'beautiful'
# print("words like " + target)
# pprint(fetchSimilar(target))