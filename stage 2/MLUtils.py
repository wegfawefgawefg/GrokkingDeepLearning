import numpy as np
import math
from PIL import Image

'''
last update:
    now supports image squaring!
    instead of cropping rectangular inputs to smaller squares. yay
'''
class MLUtils:
    @staticmethod
    def oneHot(labels):
        oneHotLabels = np.zeros((len(labels), 10))
        for i, labelIndex, in enumerate(labels):
            oneHotLabels[i][labelIndex] = 1.0
        return oneHotLabels

    @staticmethod
    def genColorImageLine(data, numImages, imageWidth):
        data = data.astype(np.uint8)
        image = Image.frombuffer(
            'RGB', (imageWidth, imageWidth*numImages), data,
            "raw", 'RGB', 0, 1)
        image = image.resize((500, 500*numImages))
        return image

    @staticmethod
    def showColorImageLineGrid(images):
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
        image = Image.frombuffer(
            'L', (imageWidth, imageWidth*numImages), data,
            "raw", 'L', 0, 1)
        image = image.resize((500, 500*numImages))
        return image

    @staticmethod
    def showImageLineGrid(images):
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
        image = Image.frombuffer(
            'L', (imageWidth, imageWidth), data,
            "raw", 'L', 0, 1)
        image = image.resize((500, 500))

    @staticmethod
    def showOneImageOfMany(data, imageNum):
        singleImage = data[imageNum]
        MLUtils.showImage(singleImage.astype(np.uint8))

    @staticmethod
    def convertLabelsToTargets(labels):
        numLabels = labels.shape[0]
        newLabels = np.zeros((numLabels, 10)).astype(np.float)
        for i, num in enumerate(labels):
            newLabels[i][num] = 1.0
        return newLabels

    @staticmethod
    def completeTheSquare(length):
        isSquare = math.floor(math.sqrt(length)) == math.sqrt(length)
        if isSquare:
            return 0
        else:
            properWidth = math.floor(math.sqrt(length)) + 1
            properLength = int(math.pow(properWidth, 2) - length)
            return properLength

    @staticmethod
    def visualizeLayers(layers):
        images = []
        for l, layer in enumerate(layers):
            #   create each grayscale image
            introspects = []
            for i in range(0, layer.shape[1]):
                introspect = layer.T[i]
                amp = 1.0
                if l == 0:
                    amp = 2000.0
                elif l == 1:
                    amp = 500.0
                introspect = introspect * amp
                introspects.append(introspect)
            introspects = np.array(introspects)

            #   pad nonsquare images
            paddingNeeded = MLUtils.completeTheSquare(introspects.shape[1])
            rightPadding = np.zeros([introspects.shape[0], paddingNeeded])
            introspects = np.hstack([introspects, rightPadding])

            #   convert to color images
            introspectsImageGreyscale = introspects.flatten()
            introspectsImage = []
            for num in introspectsImageGreyscale:
                if num > 0.0:
                    introspectsImage.append(0.0)
                    introspectsImage.append(num)
                    introspectsImage.append(0.0)
                elif num < 0.0:
                    introspectsImage.append(-num)
                    introspectsImage.append(0.0)
                    introspectsImage.append(0.0)
                elif num == 0.0:
                    introspectsImage.append(0.0)
                    introspectsImage.append(0.0)
                    introspectsImage.append(0.0)
            introspectsImage = np.asarray(introspectsImage)
            introspectsImage = np.clip(introspectsImage, 0.0, 255.0)
            imageWidth = int(math.sqrt(introspects.shape[1]))
            image = MLUtils.genColorImageLine(
                introspectsImage, layer.shape[1],  imageWidth)
            images.append(image)
        MLUtils.showColorImageLineGrid(images)

    @staticmethod
    def makeFakeImageBatch():
        a = np.arange(9)
        b = np.array([a, a*2, a*3])
        b = b.reshape(3, 3, 3)
        return b

    @staticmethod
    def getImageSection(batch, rowStart, rowEnd, colStart, colEnd):
        section = batch[:, rowStart:rowEnd, colStart:colEnd]
        return section.reshape(-1, 1, rowEnd - rowStart, colEnd - colStart)

    @staticmethod
    def kernelize2d(batch, kernelWidth):
        ''' input one or more 2d vectors 
            -   you may have to reshape before using
            -   only works on batches i believe. might not work on batch of size 1
        '''
        sections = []
        for rowStart in range(0, batch.shape[1] - kernelWidth + 1):
            for colStart in range(0, batch.shape[2] - kernelWidth + 1):
                section = MLUtils.getImageSection(
                    batch,
                    rowStart,
                    rowStart + kernelWidth,
                    colStart,
                    colStart + kernelWidth)
                sections.append(section)

        expand = np.concatenate(sections, axis=1)
        sh = expand.shape
        flat = expand.reshape(sh[0] * sh[1], -1)
        #   structure of output is:
        #   [im1sectors + im2Sectors + ...imNSectors]
        return flat, sh
