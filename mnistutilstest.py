from datasets.mnist.MNIST_UTILS import MNIST_UTILS

trainingData = MNIST_UTILS.getTrainingData()
print(trainingData[0])
trainingLabels = MNIST_UTILS.getTrainingLabels()
MNIST_UTILS.showOneImageOfMany(trainingData,7)
# print(trainingLabels[7])