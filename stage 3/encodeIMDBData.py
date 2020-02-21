from pprint import pprint
import numpy as np
import json

'''
def oneHotSomeReviews(reviews, wordToIndex):
    oneHotReviews = [oneHotReview(review, wordToIndex) for review in reviews]
    oneHotReviews = np.array(oneHotReviews)
    return oneHotReviews

def oneHotReview(review, wordToIndex):
    oneHotCrushed = np.zeros(len(wordToIndex))
    for labelIndex in review:
        oneHotCrushed[labelIndex] = 1.0
    return oneHotCrushed
'''

def fetchIMDBLabels():
    #   fetch and prepare the labels

    path = "F:\_SSD_CODEING\GrokkingDeepLearning\stage 3\labels.json"
    loaded = False
    try:
        with open(path, 'r') as inFile:
            labelsNums = json.load(inFile)
        loaded = True
        print("LOADED LABEL DATA")
    except:
        loaded = False

    if not loaded:
        f = open("labels.txt")
        rawLabels = f.readlines()
        f.close()

        labelsNums = [ 1.0 if line == 'positive\n' else 0.0 for line in rawLabels]
        with open(path, 'w') as outfile:
            json.dump(labelsNums, outfile)
        print("SAVING LABEL DATA")
        quit()

    labels = np.array(labelsNums)
    print("labels shape: " + str(labels.shape))

    return labels

def getIMDBData():
    path = "F:\_SSD_CODEING\GrokkingDeepLearning\stage 3\wordNums.json"
    path2 = "F:\_SSD_CODEING\GrokkingDeepLearning\stage 3\wordToIndex.json"
    loaded = False
    try:
        with open(path, 'r') as inFile:
            reviewWordNums = json.load(inFile)
            reviewWordNums = [np.array(review) for review in reviewWordNums]
        with open(path2, 'r') as inFile:
            wordToIndex = json.load(inFile)
        loaded = True
        print("LOADED TRAINING DATA")
    except:
        loaded = False

    if not loaded:
        print("COULDNT LOAD TRAINING DATA")

        #   create the corpus lookup and the one hot reviews
        f = open("reviews.txt")
        rawReviewsLines = f.readlines()
        f.close()

        #   #   create data set of reviews
        reviewTokenSets = [set(line.split(" ")) for line in rawReviewsLines]

        #   #   create word to index map
        corpusUncleaned = list( reviewTokenSets[0].union( *reviewTokenSets[1:] ) )
        wordToIndex = dict( zip(corpusUncleaned, range(0, len(corpusUncleaned))) )

        #   #   convert token sets to number sets
        reviewNumSets = []
        for review in reviewTokenSets:
            reviewNumSet = []
            for token in review:
                reviewNumSet.append( wordToIndex[token] )
            reviewNumSets.append( reviewNumSet )
        reviewWordNums = reviewNumSets 

        with open(path, 'w') as outfile:
            json.dump(reviewWordNums, outfile)
        with open(path2, 'w') as outfile:
            json.dump(wordToIndex, outfile)
        print("SAVING TRAINING DATA")
        quit()

    return reviewWordNums, wordToIndex