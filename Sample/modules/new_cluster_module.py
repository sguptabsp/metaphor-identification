# This file contains a second implementation of the "cluster module" which is faster than the other one


import csv
import re
import ast
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from .datastructs.metaphor_group import MetaphorGroup
from .datastructs.metaphor import Metaphor


VERBNET = "data/clustering/verbnet_150_50_200.log-preprocessed"
NOUNS = "data/clustering/200_2000.log-preprocessed"
TROFI_TAGS = "data/clustering/trofi_tags_full.csv"

LABELED_VECTORS_PATH = "./data/clustering/DB/labeled_vectors/"
UNLABELED_VECTORS_PATH = "./data/clustering/DB/unlabeled_vectors/"
VECTORS_PATHS = {'labeledVectors': LABELED_VECTORS_PATH, 'unlabeledVectors': UNLABELED_VECTORS_PATH}

# cluster2verb maps a cluster (integer) to a list of verbs
def parseVerbClusterFile(file):
    cluster2verb = dict()

    with open(file, 'r') as file:
        lines = file.readlines()
        name = ""
        content = []
        for l in lines:
            if l.startswith(" <"):
                name = re.findall(r'"([^"]*)"', l)[0]
            elif l.startswith("-"):
                continue
            else:
                content = l.split()
                cluster2verb[name] = content

    return cluster2verb


# verb2cluster maps a verb to a cluster (integer)
def createVerbClustersDatastruct(path):
    cluster2verb = parseVerbClusterFile(path)

    verb2cluster = dict()
    for cluster, verbs in cluster2verb.items():
        for verb in verbs:
            verb2cluster[verb] = cluster

    return cluster2verb, verb2cluster


# cluster2noun maps a cluster (integer) to a list of nouns
def parseNounClusterFile(file):
    cluster2noun = dict()

    with open(file, 'r') as file:
        lines = file.readlines()

        for l in lines:
            newClusterContent = {}
            newClusterContent["words"] = []

            wordsInLine = l.split()
            name = wordsInLine[0][7:]
            content = wordsInLine[1:]

            cluster2noun[name] = content

    return  cluster2noun


# noun2cluster maps a noun to a cluster (integer)
def createNounClustersDatastruct(path):
    cluster2noun = parseNounClusterFile(path)

    noun2cluster = dict()
    for cluster, nouns in cluster2noun.items():
        for noun in nouns:
            noun2cluster[noun] = cluster

    return cluster2noun, noun2cluster


# Get the  tags from a CSV file (trofi full)
def getTagsFromCSV(path):
    verbObjTags = {}
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            verb = row["Verb"]
            noun = row["Noun"]
            verbObjTags[(verb, noun)] = ast.literal_eval(row["Labels"])
    return verbObjTags


# cluster2label maps a pair of clusters
def createLabelClustersDatastruct(tagsPath=TROFI_TAGS, verbClustersPath=VERBNET, nounClustersPath=NOUNS):
    cluster2label = dict()

    labeledPairs = getTagsFromCSV(tagsPath)
    cluster2verb, verb2cluster = createVerbClustersDatastruct(verbClustersPath)
    cluster2noun, noun2cluster = createNounClustersDatastruct(nounClustersPath)

    countUselessPairs = 0

    for pair, labels in labeledPairs.items():
        verbCluster = verb2cluster.get(pair[0], -1)
        nounCluster = noun2cluster.get(pair[1], -1)

        if verbCluster == -1 or nounCluster == -1:
            countUselessPairs += 1
            continue

        L = cluster2label.get((verbCluster, nounCluster), [])
        L.extend(labels)
        cluster2label[(verbCluster, nounCluster)] = L


    for pair, labels in cluster2label.items():
        nLabels = len(labels)
        countLiteral = labels.count('L')
        countMetaphorical = nLabels - countLiteral

        literalConfidence = countLiteral / nLabels
        metaphoricalConfidence = countMetaphorical / nLabels

        if literalConfidence > 0.5:
            cluster2label[pair] = ("L", literalConfidence)
        else:
            cluster2label[pair] = ("N", metaphoricalConfidence)

    print('Number of useless pairs', countUselessPairs, 'out of', len(labeledPairs.keys()), 'seed pairs')
    print('Number of cluster relationships', len(cluster2label.keys()))

    return cluster2label


def loadDB():
    DB = dict()
    print("Create cluster2verb")
    DB['cluster2verb'], DB['verb2cluster'] = createVerbClustersDatastruct(VERBNET)
    print("Create cluster2noun")
    DB['cluster2noun'], DB['noun2cluster'] = createNounClustersDatastruct(NOUNS)
    print("Create cluster2label")
    DB['cluster2label'] = createLabelClustersDatastruct(TROFI_TAGS, VERBNET, NOUNS)
    DB['labeledVectors'] = dict()
    DB['unlabeledVectors'] = dict()

    for key, path in VECTORS_PATHS.items():
        print("Load", key)
        filenames = os.listdir(path)
        for f in filenames:
            file = open(path + f, "rb")
            data = pickle.load(file)
            file.close()
            DB[key][f[0]] = data

    return DB


def getVector(DB, word):
    firstLetter = word[0]
    if word in DB['labeledVectors'][firstLetter].keys():
        return DB['labeledVectors'][firstLetter][word]
    elif word in DB['unlabeledVectors'][firstLetter].keys():
        return DB['unlabeledVectors'][firstLetter][word]


def getWordCluster(word, DB, pos):
    if pos == 'verb':
        id = ("verb2cluster", "cluster2verb")
    elif pos == 'noun':
        id = ("noun2cluster", "cluster2noun")

    wordCluster = DB[id[0]].get(word, -1)

    if wordCluster != -1:
        return wordCluster

    # Find a cluster where the words are similar to verb
    cluster2averageSimilarity = dict()

    try:
        wordVector = getVector(DB, word).reshape(1, -1)
    except:
        return -1

    for cluster, words in DB[id[1]].items():
        similarities = list()
        for otherWord in words:
            try:
                otherWordVector = getVector(DB, otherWord).reshape(1, -1)
            except:
                continue
            similarities.append(cosine_similarity(wordVector, otherWordVector)[0][0])
        if similarities != []:
            cluster2averageSimilarity[cluster] = sum(similarities) / float(len(similarities))

    c = -1
    max_sim = -1
    for cluster, sim in cluster2averageSimilarity.items():
        if sim > max_sim:
            c = cluster
            max_sim = sim

    return c


def getResult(sourceCluster, targetCluster, DB):
    pair = (sourceCluster, targetCluster)

    if pair in DB['cluster2label'].keys():
        return DB['cluster2label'][pair]
    else:
        # print("Don't know")
        return (False, 0.0)


def newClusterModule(candidates, cand_type, verbose):
    # 1. Use verb2cluster to find the cluster of the verb
    # 2. If no cluster
    #   1. use cluster2verb to find the cluster in which the verb could fit
    #   2. Use this cluster
    # 3. Use noun2cluster to find th cluster of the noun
    # 4. If no cluster
    #   1. use cluster2noun to find the cluster in which the noun could fit
    #   2. use this cluster
    # 5. Use clusters2label to return the label of the pair

    results = MetaphorGroup()
    DB = loadDB()

    for c in candidates:
        source = c.getSource()
        target = c.getTarget()

        sourceCluster = getWordCluster(source, DB, 'verb')
        targetCluster = getWordCluster(target, DB, 'noun')

        # if sourceCluster != -1:
            # print(source)
            # print(DB['cluster2verb'][sourceCluster])
        # if targetCluster != -1:
            # print(target)
            # print(DB['cluster2noun'][targetCluster])

        result = getResult(sourceCluster, targetCluster, DB)
        label = (result[0] == "N") # Assign True to label if Non-Literal, False otherwise
        confidence = result[1]
        # print('')

        results.addMetaphor(Metaphor(c, label, confidence))

    return results

if __name__ == '__main__':
    import pandas as pd