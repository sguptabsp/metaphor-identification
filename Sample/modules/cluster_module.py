from .clustering.wordClusterList import WordClusterList
from .clustering.parsing import parseNouns, parseVerbNet
from .utils import writeToCSV
from .datastructs.metaphor_group import MetaphorGroup
from .datastructs.metaphor import Metaphor
import csv
import pandas as pd
import ast
import pickle
from gensim.models import KeyedVectors

from tqdm import tqdm

VERBNET = "data/clustering/verbnet_150_50_200.log-preprocessed"
NOUNS = "data/clustering/200_2000.log-preprocessed"
TROFI_TAGS = "data/clustering/trofi_tags_full.csv"

RESULTS = "data/clustering/results2.csv"
# RESULTS = "../data/clustering/results.csv"

# test the result to see if it has every words
# Parse the verbnet and noun databases
def getVerbNouns(verbPath, nounPath):
    verbData = WordClusterList.fromFile(verbPath, parseVerbNet)
    #print(data1)

    nounData = WordClusterList.fromFile(nounPath, parseNouns)
    #print(data2)

    return [verbData, nounData]

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

# Get a list of all the possible verb-noun couples in the database, returns a list of labeled_pairs of 2 lists, one list of verbs from the same cluster and one list of nouns from the same cluster
def buildPairs(verbsData, nounsData):
    return [(v, n) for v in verbsData for n in nounsData]


# Main algorithm
def tagPairs(verbsData, nounsData, tags):
    pairs = buildPairs(verbsData, nounsData)
    results = []
    for p in pairs:
        literals = 0
        nonliterals = 0
        verbNounPairs = []
        for verb in p[0]:
            for noun in p[1]:
                currentPair = (verb, noun)
                verbNounPairs.append(currentPair)
                if currentPair in tags.keys():
                    for t in tags[currentPair]:
                        if t == "N":
                            nonliterals+=1
                        elif t == "L":
                            literals+=1
        total = literals+nonliterals
        if total != 0:
            for vn in verbNounPairs:
                currentResult = {}
                currentResult["verb"] = vn[0]
                currentResult["noun"] = vn[1]
                if literals/total > 0.5:
                    currentResult["tag"] = "L"
                    currentResult["confidence"] = literals/total
                else:
                    currentResult["tag"] = "N"
                    currentResult["confidence"] = nonliterals/total
                results.append(currentResult)
        else:
            for vn in verbNounPairs:
                currentResult = {}
                currentResult["verb"] = vn[0]
                currentResult["noun"] = vn[1]
                currentResult["tag"] = "L"
                currentResult["confidence"] = 0.1
                results.append(currentResult)

    return results

# Older DB Format
# def buildDB():
# 	verbs, nouns = getVerbNouns(VERBNET, NOUNS)
# 	# pairs = buildPairs(verbs, nouns)
# 	tags = getTagsFromCSV(TROFI_TAGS)
# 	results = tagPairs(verbs, nouns, tags)
# 	writeToCSV(results, RESULTS, ["verb", "noun", "tag", "confidence"])


def loadDB(path):
    DB = {}
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            DB[(row["verb"], row["noun"])] = (row["tag"], float(row["confidence"]))
    return DB


def lookUpDB(DB, verb, noun):
    if (verb, noun) in DB.keys():
        result = DB[(verb, noun)]
        if result[0] == "L":
            return (False, result[1])
        else:
            return (True, result[1])
    else:
        return (False, 0.0)


# MAIN FUNCTION
def clusteringFunction(candidates, cand_type, verbose):

    results = MetaphorGroup()
    DB = loadDB(RESULTS)

    for c in candidates:
        source = c.getSource()
        target = c.getTarget()
        if verbose:
            print("###############################################################################")
            print("SOURCE: " + source)
            print("TARGET: " + target)

        currentResult = lookUpDB(DB, source, target)
        result = currentResult[0]
        confidence = currentResult[1]

        if verbose:
            if confidence >= 0.5:
                print("RESULT: " + str(result))
                print("CONFIDENCE: " + str(confidence))
            else:
                print("RESULT: Unknown")
                print("Verb/Noun pair not in database")

        results.addMetaphor(Metaphor(c, result, confidence))

    return results


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #


LABELED_PAIRS_PATH = "./data/clustering/DB/labeled_pairs/"

LABELED_VECTORS_PATH = "./data/clustering/DB/labeled_vectors/"
UNLABELED_VECTORS_PATH = "./data/clustering/DB/unlabeled_vectors/"
VECTORS_PATHS = {'labeledVectors': LABELED_VECTORS_PATH, 'unlabeledVectors': UNLABELED_VECTORS_PATH}



def formatResults(results):
    DB = dict()
    for row in results:
        firstLetter = row['verb'][0].lower()
        tempDict = DB.get(firstLetter, dict())
        tempDict[(row["verb"], row["noun"])] = (row["tag"], float(row["confidence"]))
        DB[firstLetter] = tempDict
    return DB


def writeOnDisk(DB):
    import pickle
    for letter, dic in DB.items():
        file = open(LABELED_PAIRS_PATH + letter + ".pickle", "wb")
        pickle.dump(dic, file)
        file.close()


def buildPairDB():
    verbs, nouns = getVerbNouns(VERBNET, NOUNS)
    tags = getTagsFromCSV(TROFI_TAGS)
    results = tagPairs(verbs, nouns, tags)
    DB = formatResults(results)
    writeOnDisk(DB)


def buildWordVectorDB():
    filename = 'GoogleNews-vectors-negative300.bin'
    vectors = KeyedVectors.load_word2vec_format(filename, binary=True)
    words = list(vectors.wv.vocab)

    file_names_cluster_DB = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'y']
    path = './DB/labeled_pairs/'

    verbs = dict()
    nouns = dict()
    verb_list = list()
    noun_list = list()

    # Storing verbs in dictionary
    # Storing one big list of nouns
    for l in tqdm(file_names_cluster_DB):
        df = pd.read_csv(path + l + '.csv')
        v = list(set(df['verb'].values.tolist()))
        verbs[l] = v
        verb_list.extend(v)
        noun_list.extend(list(set(df['noun'].values.tolist())))

    # Keeping nouns only once
    noun_list = list(set(noun_list))

    for n in tqdm(noun_list):
        first_letter = n[0]
        l = nouns.get(first_letter, list())
        l.append(n)
        nouns[first_letter] = l

    verb_vectors = dict()

    # For each first letter
    for first_letter, verb_list in verbs.items():
        vector_dict = dict()

        # for each verb
        for v in tqdm(verb_list):
            if v in words:
                vector_dict[v] = vectors[v]
            else:
                print(v)

        verb_vectors[first_letter] = vector_dict

    noun_vectors = dict()

    # For each first letter
    for first_letter, noun_list in nouns.items():
        vector_dict = dict()

        # for each verb
        for n in tqdm(noun_list):
            if n in words:
                vector_dict[n] = vectors[n]
            else:
                print(n)

        noun_vectors[first_letter] = vector_dict

    import pickle

    path = './DB/labeled_vectors/'

    for letter, vec_dict in verb_vectors.items():
        file = open(path + letter + "_verbs" + ".pickle", "wb")
        pickle.dump(vec_dict, file)
        file.close()

    for letter, noun_dict in noun_vectors.items():
        file = open(path + letter + "_nouns" + ".pickle", "wb")
        pickle.dump(noun_dict, file)
        file.close()

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    alphabetical_vector_dict = dict()

    for w in tqdm(words):
        first_letter = w[0]

        # Skip the words that are not in the alphabet or that are in the cluster DB
        if first_letter not in alphabet:
            continue
        if w in verb_list:
            continue
        if w in noun_list:
            continue

        d = alphabetical_vector_dict.get(first_letter, dict())
        d[w] = vectors[w]
        alphabetical_vector_dict[first_letter] = d

    path = './DB/unlabeled_vectors/'

    for first_letter, vector_dict in tqdm(alphabetical_vector_dict.items()):
        file = open(path + first_letter + ".pickle", "wb")
        pickle.dump(vector_dict, file)
        file.close()


def buildDB():
    buildPairDB()
    buildWordVectorDB()


def loadVectors(firstLetter):
    data = dict()
    for type, path in VECTORS_PATHS:
        file = open(path + firstLetter + ".pickle", "rb")
        data[type] = pickle.load(file)
        file.close()
    return data


def loadLabeledPairs(firstLetter):
    file = open(LABELED_PAIRS_PATH + firstLetter + ".pickle", "rb")
    data = pickle.load(file)
    file.close()
    return data


def initDB():
    import os

    DB = dict()
    DB['labeledPairs'] = dict()
    DB['labeledVectors'] = dict()
    DB['unlabeledVectors'] = dict()
    DB['similarWords'] = dict() # Used as a cache

    for key, path in VECTORS_PATHS.items():
        filenames = os.listdir(path)
        for f in filenames:
            file = open(path + f, "rb")
            data = pickle.load(file)
            file.close()
            DB[key][f[0]] = data

    return DB


#type is either "pair" or "vectors"
def updateDB(DB, firstLetter, type="pairs"):

    if type == "pairs":
        if firstLetter not in DB['labeledPairs'].keys():
            DB['labeledPairs'][firstLetter] = loadLabeledPairs(firstLetter)

    elif type == "vectors":
        data = loadVectors(firstLetter)
        for type, vectors in data:
            if firstLetter not in DB[type].keys():
                DB[type][firstLetter] = vectors

    return DB


def loadAllDB(DB):
    import os
    filenames = os.listdir(LABELED_PAIRS_PATH)
    for f in filenames:
        firstLetter = f[0]
        DB = updateDB(DB, firstLetter)
    return DB


def getVector(DB, word):
    firstLetter = word[0]
    if word in DB['labeledVectors'][firstLetter].keys():
        return DB['labeledVectors'][firstLetter][word]
    elif word in DB['unlabeledVectors'][firstLetter].keys():
        return DB['unlabeledVectors'][firstLetter][word]


# Pick the most similar word to word in the list otherWords
def getMostSimilarWord(DB, word, otherWords):

    if word in DB['similarWords'].keys():
        print('--------------------------------- Already did the calculation')
        return DB['similarWords'][word]

    from sklearn.metrics.pairwise import cosine_similarity

    wordVector = getVector(DB, word)
    wordVector = wordVector.reshape(1, -1)

    similarity = -1

    for ow in otherWords:
        owVector = getVector(DB, ow)
        try:
            owVector = owVector.reshape(1, -1)
        except: # The word might not have a vector available
            continue
        s = cosine_similarity(wordVector, owVector)[0][0]
        if s > similarity:
            mostSimilarWord = ow
            similarity = s

    DB['similarWords'][word] = (mostSimilarWord, similarity)
    return mostSimilarWord, similarity


# Return the most similar noun that is paired with verb
def getSimilarNounPairedToVerb(DB, verb, noun):
    print('getSimilarNounPairedToVerb()')

    verbFirstLetter = verb[0]
    nouns = [pair[1] for pair in DB['labeledPairs'][verbFirstLetter].keys() if pair[0] == verb]

    nouns = list(set(nouns))
    mostSimilarNoun, similarity = getMostSimilarWord(DB, noun, nouns)
    return mostSimilarNoun, similarity


# Return the most similar verb that is paired with noun
def getSimilarVerbPairedToNoun(DB, verb, noun):
    print('getSimilarVerbPairedToNoun()')

    allVerbs = list()
    letters = DB['labeledPairs'].keys()

    for l in letters:
        allVerbs.extend([pair[0] for pair in DB['labeledPairs'][l].keys() if pair[1] == noun])

    allVerbs = list(set(allVerbs))
    mostSimilarVerb, similarity = getMostSimilarWord(DB, verb, allVerbs)
    return mostSimilarVerb, similarity


# Return a verb in the DB that is similar to verb
def getSimilarVerb(DB, verb):
    print('getSimilarVerb()')
    # DB = loadAllDB(DB)

    allVerbs = list()
    letters = DB['labeledPairs'].keys()

    for l in letters:
        allVerbs.extend([pair[0] for pair in DB['labeledPairs'][l].keys()])

    allVerbs = list(set(allVerbs))
    mostSimilarVerb, similarity = getMostSimilarWord(DB, verb, allVerbs)
    return mostSimilarVerb, similarity


# Return True if verb is in the DB
def verbInDB(DB, verb):
    verbFirstLetter = verb[0].lower()
    verbsInFirstLetter = [pair[0] for pair in DB['labeledPairs'][verbFirstLetter].keys()] # List of verbs which start with the same letter as verb
    return verb in verbsInFirstLetter


# Return True if noun is in the DB
def nounInDB(DB, noun):
    DB = loadAllDB(DB)
    letters = DB['labeledPairs'].keys()

    for l in letters:
        nounsInDB = [pair[1] for pair in DB['labeledPairs'][l].keys()]
        if noun in nounsInDB:
            return True

    return False


# Return True if the pair is in the DB
def pairInDB(DB, verb, noun):
    return (verb, noun) in DB['labeledPairs'].keys()


# Return a 2-tuple (Label, Confidence) where Label is a boolean and Confidence a number between 0 and 1
def getLabelConfidence(DB, verb, noun, coeff=1):
    result = DB['labeledPairs'][verb[0]][(verb, noun)]

    if result[0] == "L":
        print('Label = False\n')
        return (False, coeff * result[1])
    else:
        print('Label = True')
        return (True, coeff * result[1])


# ALGORITHM:
# If the pair of words is in the DB, then return the label and confidence
# Else
#   If the verb is in the DB, but the noun is either not in the DB or not paired with this verb in the DB
#     Find a noun that is similar and paired with this verb, use this new pair and multiply the confidence by the similarity
#   If the verb is not in the DB
#     If the noun is in the DB
#       Find a verb that is similar and paired with this noun, use this new pair and multiply the confidence by the similarity
#     Else
#       Find a verb in the DB which is similar to the original verb
#       Find a noun in the DB which is similar to the original noun and paired with the similar verb
#       Use this new pair and multiply the confidence by the product of the 2 similarities
# END ALGORITHM
# If a word does not have a vector, then the exception is handled by returning (label=False, confidence=0.0)
def getResult(DB, verb, noun):

    if pairInDB(DB, verb, noun):
        return getLabelConfidence(DB, verb, noun)

    else:

        if verbInDB(DB, verb):
            try:
                similarNoun, similarity = getSimilarNounPairedToVerb(DB, verb, noun)
            except: # It means that noun does not have a vector
                print('Did not find a vector for', noun, '\n')
                return (False, 0.0)
            print('new pair = (', verb, ',', similarNoun, '), coeff =,', similarity)
            return getLabelConfidence(DB, verb, similarNoun, coeff=similarity)

        else:
            if nounInDB(DB, noun):
                DB = loadAllDB(DB)
                try:
                    similarVerb, similarity = getSimilarVerbPairedToNoun(DB, verb, noun)
                except: # It means that verb does not have a vector
                    print('Did not find a vector for', verb, '\n')
                    return (False, 0.0)
                print('new pair = (', similarVerb, ',', noun, '), coeff =,', similarity)
                return getLabelConfidence(DB, similarVerb, noun, coeff=similarity)

            else:
                DB = loadAllDB(DB)
                try:
                    similarVerb, similarityVerb = getSimilarVerb(DB, verb)
                except: # It means that verb does not have a vector
                    print('Did not find a vector for', verb, '\n')
                    return (False, 0.0)

                try:
                    similarNoun, similarityNoun = getSimilarNounPairedToVerb(DB, similarVerb, noun)
                except: # It means that noun does not have a vector
                    print('Did not find a vector for', noun, '\n')
                    return (False, 0.0)

                print('new pair = (', similarVerb, ',', similarNoun, '), coeff =,', similarityVerb*similarityNoun)
                return getLabelConfidence(DB, similarVerb, similarNoun, coeff=similarityVerb*similarityNoun)


# MAIN FUNCTION WITH WORD VECTORS
def clusteringFunction_2(candidates, cand_type, verbose):

    results = MetaphorGroup()
    print('Init DB...')
    DB = initDB()
    # path = './data/clustering/DB/labeled_pairs/'

    for c in candidates:
        source = c.getSource()
        target = c.getTarget()
        print('(', source, ',', target, ')')

        if verbose:
            print("###############################################################################")
            print("SOURCE: " + source)
            print("TARGET: " + target)

        firstLetter = source[0].lower()
        DB = updateDB(DB, firstLetter, type="pairs")

        currentResult = getResult(DB, source, target) # Main algorithm
        result = currentResult[0]
        confidence = currentResult[1]

        if verbose:
            if confidence >= 0.5:
                print("RESULT: " + str(result))
                print("CONFIDENCE: " + str(confidence))
            else:
                print("RESULT: Unknown")
                print("Verb/Noun pair not in database")

        results.addMetaphor(Metaphor(c, result, confidence))

    return results

