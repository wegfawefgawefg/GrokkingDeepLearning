#   create the corpus
f = open("reviews.txt")
rawReviewsLines = f.readlines()
f.close()

#   use the corpus to create the list of reviews with word vectors
reviewTokenSets = [set(line.split(" ")) for line in rawReviewsLines]
corpusSetUncleaned = list( reviewTokenSets[0].union( *reviewTokenSets[1:] ) )
corpusSetUncleaned.sort()
print(corpusSetUncleaned)


#   collapse the review one hot vectors