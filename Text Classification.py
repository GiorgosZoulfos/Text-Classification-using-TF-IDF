import os
import math
import operator
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from operator import itemgetter
from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm

stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
regex = re.compile('[^a-zA-Z ]')
N = 4000  # number of features


def processFiles(filePath):
    processedFile = []

    with open(filePath, 'r') as file:
        for line in file.read().splitlines():
            if ':' not in line:
                processedFile.append(line)

    processedFile = str(processedFile)
    processedFile = regex.sub('', processedFile)
    wordArr = word_tokenize(processedFile)
    wordArr = [stemmer.stem(word) for word in wordArr]
    wordArr = [word for word in wordArr if word not in stopwords]

    return wordArr


def computeTfIdf(files):
    tfDic = {}
    idfDic = {}
    allWords = {}
    tf_idf = []
    fileId = 0

    for file in files:
        numberOfWords = len(file)
        wordFreq = {}

        for word in file:
            if word in wordFreq:
                wordFreq[word] += 1
            else:
                wordFreq[word] = 1

        for word in wordFreq:
            wordFreq[word] = wordFreq[word] / numberOfWords

            if word in allWords:
                allWords[word] += 1
            else:
                allWords[word] = 1

        tfDic[fileId] = wordFreq
        fileId += 1

    for word in allWords:
        idfDic[word] = math.log10(len(files) / allWords[word])

    for page in tfDic:
        metric = {}
        for word in tfDic[page]:
            metric[word] = tfDic[page][word] * idfDic[word]

        tf_idf.append(metric)

    return tf_idf


def readFiles(path):
    entries = os.listdir(path)

    files = []
    labels = []

    for i in entries:
        fullPath = path + i
        moreEntries = os.listdir(fullPath)

        for j in moreEntries:
            pathToFile = fullPath + '/' + j
            curFile = processFiles(pathToFile)

            files.append(curFile)
            labels.append(i)

    return files, labels


def createFeatureVector(tfifd):
    allWords = {}
    for table in tfifd:
        allWords.update(table)

    featureVector = {}
    for i in range(1, N):
        cur_max = max(allWords.items(), key=operator.itemgetter(1))

        featureVector[cur_max[0]] = cur_max[1]
        del allWords[cur_max[0]]

    return featureVector


def createVectors(tfidf, feat):
    vectors = []
    for file in tfidf:
        vector = []
        for word in feat:
            if word in file:
                vector.append(feat[word])
            else:
                vector.append(0)
        vectors.append(vector)
    return vectors


def cosineSimilarity(vec1, vec2):
    return distance.cosine(vec1, vec2)



def jaccardDistance(vec1, vec2):
    return distance.jaccard(vec1, vec2)


# -------  main begins  -------

trainFiles, trainLabels = readFiles('train2/')
testFiles, testLabels = readFiles('test2/')

tfidfE = computeTfIdf(trainFiles)
tfidfA = computeTfIdf(testFiles)

features = createFeatureVector(tfidfE)

trainVec = createVectors(tfidfE, features)
testVec = createVectors(tfidfA, features)


estimatedLabels = []
for vec in testVec:
    sim = []
    for tvec in trainVec:
        # sim.append(cosineSimilarity(vec, tvec))
        sim.append(jaccardDistance(vec, tvec))

    indexOfMax = sim.index(min(sim))
    # print(indexOfMax)
    estimatedLabels.append(trainLabels[indexOfMax])

counter = 0
for i in range(len(testLabels)):
    if testLabels[i] == estimatedLabels[i]:
        counter += 1

print((counter/len(estimatedLabels))*100)

# dict(sorted(tf_idf.items(), key=itemgetter(1), reverse=True)[:N])