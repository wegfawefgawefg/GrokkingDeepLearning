'''
Exposes MNIST dataset as functions for easy fetching of labels and whatnot
'''

import numpy as np
import os

class MNIST_UTILS:
    @staticmethod
    def fetchLabels(fileName):
        f = open(fileName, 'rb')

        headerBuf = f.read(2*4)
        hdt = np.dtype(np.int32)
        hdt = hdt.newbyteorder(">")
        header = np.frombuffer(headerBuf, dtype=hdt)
        numLabels = header[1]
        print("numLabels: " + str(numLabels))

        ddt = np.dtype(np.uint8)
        ddt = ddt.newbyteorder(">")
        buf = f.read(numLabels)
        data = np.frombuffer(buf, dtype=ddt).astype(np.uint8)

        return data

    @staticmethod
    def fetchData(fileName, normalized=False):
        image_size = 28
        f = open(fileName, 'rb')

        headerBuf = f.read(4*4)
        hdt = np.dtype(np.int32)
        hdt = hdt.newbyteorder(">")
        header = np.frombuffer(headerBuf, dtype=hdt)
        numImages = header[1]
        print("numImages: " + str(numImages))

        ddt = np.dtype(np.uint8)
        ddt = ddt.newbyteorder(">")
        buf = f.read(image_size * image_size * numImages)
        data = np.frombuffer(buf, dtype=ddt).astype(np.float)
        if normalized:
            if os.path.isfile('normalizedMnist.inputs.npy'):
                print("saved mnist normalized data found! loading saved...")
                data = np.load('normalizedMnist.inputs.npy')
            else:
                print("no saved mnist normalized data found!")
                print("normalizing Data...")
                from sklearn.preprocessing import scale
                data = scale( data, axis=0, with_mean=True, with_std=True, copy=True )
                np.save('normalizedMnist.inputs', data)
                print("Complete!")
        data = data.reshape(numImages, image_size*image_size)
        return data

    @staticmethod
    def getTrainingData(normalized=False):
        print("Fetching Training Dataset:")
        fileName = "H:/Coding/Python/1_TrueFluency/GrokkingDeep/datasets/mnist/train-images.idx3-ubyte"
        return MNIST_UTILS.fetchData(fileName, normalized)
    
    @staticmethod
    def getTrainingLabels():
        print("Fetching Training Dataset Labels:")
        fileName = "H:/Coding/Python/1_TrueFluency/GrokkingDeep/datasets/mnist/train-labels.idx1-ubyte"
        return MNIST_UTILS.fetchLabels(fileName)
    
    @staticmethod
    def getTestingData():
        print("Fetching Testing Dataset:")
        fileName = "H:/Coding/Python/1_TrueFluency/GrokkingDeep/datasets/mnist/t10k-images.idx3-ubyte"
        return MNIST_UTILS.fetchData(fileName)
    
    @staticmethod
    def getTestingLabels():
        print("Fetching Testing Dataset Labels:")
        fileName = "H:/Coding/Python/1_TrueFluency/GrokkingDeep/datasets/mnist/t10k-labels.idx1-ubyte"
        return MNIST_UTILS.fetchLabels(fileName)

    @staticmethod
    def genColorImageLine(data, numImages, imageWidth):
        data = data.astype(np.uint8)
        from PIL import Image
        image = Image.frombuffer(
            'RGB', (imageWidth, imageWidth*numImages), data,
            "raw", 'RGB', 0, 1)
        image = image.resize((500, 500*numImages))
        return image

    @staticmethod
    def showColorImageLineGrid(images):
        from PIL import Image
        imageWidth = len(images)*500
        greatestHeight = max([image.size[1] for image in images])
        compositeImage = Image.new('RGB', (imageWidth, greatestHeight))
        for i, image in enumerate(images):
            compositeImage.paste(image, (i*500, 0))
        compositeImage.save("compImage.png", "PNG")
        compositeImage.show()

    @staticmethod
    def genImageLine(data, numImages, imageWidth):
        data = data.astype(np.uint8)
        from PIL import Image
        image = Image.frombuffer(
            'L', (imageWidth, imageWidth*numImages), data,
            "raw", 'L', 0, 1)
        image = image.resize((500, 500*numImages))
        return image

    @staticmethod
    def showImageLineGrid(images):
        from PIL import Image
        imageWidth = len(images)*500
        greatestHeight = max([image.size[1] for image in images])
        compositeImage = Image.new('L', (imageWidth, greatestHeight))
        for i, image in enumerate(images):
            compositeImage.paste(image, (i*500, 0))
        compositeImage.save("compImage.png", "PNG")
        compositeImage.show()

    @staticmethod
    def genImage(data, imageWidth):
        data = data.astype(np.uint8)
        from PIL import Image
        image = Image.frombuffer(
            'L', (imageWidth, imageWidth), data,
            "raw", 'L', 0, 1)
        image = image.resize((500, 500))

    @staticmethod
    def showOneImageOfMany(data, imageNum):
        singleImage = data[imageNum]
        MNIST_UTILS.showImage(singleImage.astype(np.uint8))

    @staticmethod
    def convertLabelsToTargets(labels):
        numLabels = labels.shape[0]
        newLabels = np.zeros((numLabels, 10)).astype(np.float)
        for i, num in enumerate(labels):
            newLabels[i][num] = 1.0
        return newLabels
