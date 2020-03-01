import numpy as np

def softmax(x_):
    x = np.atleast_2d(x_)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

wordVecs = {}
wordVecs['yankees'] = np.array([[0.,0.,0.]])
wordVecs['bears'] = np.array([[0.,0.,0.]])
wordVecs['braves'] = np.array([[0.,0.,0.]])
wordVecs['red'] = np.array([[0.,0.,0.]])
wordVecs['socks'] = np.array([[0.,0.,0.]])
wordVecs['lose'] = np.array([[0.,0.,0.]])
wordVecs['defeat'] = np.array([[0.,0.,0.]])
wordVecs['beat'] = np.array([[0.,0.,0.]])
wordVecs['tie'] = np.array([[0.,0.,0.]])

weights = np.random.rand(3,len(wordVecs))

identity = np.eye(3)

y = np.array([1,0,0,0,0,0,0,0,0]) # target one-hot vector for "yankees"

#   #   phrase: red socks defeat yankees
layer0Output = wordVecs['red']
layer1Output = layer0Output.dot(identity) + wordVecs['socks']
layer2Output = layer1Output.dot(identity) + wordVecs['defeat']
layer3Output = softmax( layer2Output.dot(weights) )

#   to channel back the gradient through a pipe, its gradient.dot(stationary)
#   #   for scenario pipe.dot(stationary)
layer3Delta = layer3Output - y
layer2Delta = layer3Delta.dot(weights.T)
defeatDelta = layer2Delta
#   one of the identity deltas is here
layer1Delta = layer2Delta.dot(identity.T)
socksDelta = layer1Delta
#   one of the identity deltas is here
layer0Delta = layer1Delta.dot(identity.T)
redDelta = layer0Delta

#   now update the weights
alpha = 0.01
wordVecs['red'] -= alpha * redDelta
wordVecs['socks'] -= alpha * socksDelta
wordVecs['defeat'] -= alpha * defeatDelta
identity -= alpha * np.outer( layer1Output, layer2Delta)
identity -= alpha * np.outer( layer0Output, layer1Delta)
