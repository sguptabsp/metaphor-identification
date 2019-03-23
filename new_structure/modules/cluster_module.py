from .clustering.wordClusterList import WordClusterList
from .clustering.parsing import parseNouns, parseVerbNet
from .utils import writeToCSV
from .datastructs.labeled_metaphor_list import LabeledMetaphorList
from .datastructs.labeled_metaphor import LabeledMetaphor
import csv
import ast

VERBNET = "data/clustering/verbnet_150_50_200.log-preprocessed"
NOUNS = "data/clustering/200_2000.log-preprocessed"
TROFI_TAGS = "data/clustering/trofi_tags_full.csv"

RESULTS = "data/clustering/results.csv"
# RESULTS = "../data/clustering/results.csv"

# test the result to see if it has every words
# Parse the verbnet and noun databases
def getVerbNouns(verbPath, nounPath):
	verbData = WordClusterList.fromFile(verbPath, parseVerbNet)
	#print(data1)
	#
	nounData = WordClusterList.fromFile(nounPath, parseNouns)
	#print(data2)
	#
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

# Get a list of all the possible verb-noun couples in the database, returns a list of pairs of 2 lists, one list of verbs from the same cluster and one list of nouns from the same cluster
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
	return results

def buildDB():
	verbs, nouns = getVerbNouns(VERBNET, NOUNS)
	# pairs = buildPairs(verbs, nouns)
	tags = getTagsFromCSV(TROFI_TAGS)
	results = tagPairs(verbs, nouns, tags)
	writeToCSV(results, RESULTS, ["verb", "noun", "tag", "confidence"])

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

	results = LabeledMetaphorList()
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

		results.addResult(LabeledMetaphor(c, result, confidence))

	return results