from modules.datastructs.clustered_data import ClusteredData
from modules.parsing_functions import parseVerbNet, parseNouns, parseTroFi
from modules.utils import writeToCSV
from modules.cluster_module import buildDB
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import sys
import ast
import csv

FIRST_CLUSTER = 2
LAST_CLUSTER = 3
BUILD_TROFI = False

# Find the object of a verb in a sentence
def getObj(sentence, verb):
	parser = StanfordDependencyParser()
	lemmatizer = WordNetLemmatizer()
	dependency_tree = [list(line.triples()) for line in parser.raw_parse(sentence)]
	dependencies = dependency_tree[0]
	verbLemma = lemmatizer.lemmatize(verb, wordnet.VERB)
	obj = ""
	for dep in dependencies:
		if  "VB" in dep[0][1]:
			depVerbLemma = lemmatizer.lemmatize(dep[0][0], wordnet.VERB)
			if (verbLemma == depVerbLemma and ("obj" in dep[1] or "nsubjpass" in dep[1])):
				obj = dep[2][0] #lemmatize the noun

	return lemmatizer.lemmatize(obj, wordnet.NOUN) 

# Determine the labels of the verb-object relationship (L for litteral, N for non-litteral)
# Returns a dictionnary with the verb-object tuple as the key and the list of labels found as the value
def getVerbObjTags(data, first, last):
	verbObjTags = {}
	objects = []
	verbs = []
	dataEntries = []
	if first == 0 and last == 0:
		dataEntries = data.getEntries()
	elif last == 0:
		dataEntries = data.getEntries()[first:]
	else:
		dataEntries = data.getEntries()[first:last+1]
	for verb in dataEntries:
		actualVerb = ""
		clusterTag = ""
		if verb.endswith("-NL"):
			actualVerb = verb[:-3]
			clusterTag = "N"
		elif verb.endswith("-L"):
			actualVerb = verb[:-2]
			clusterTag = "L"

		clusterSentences = data.getClusterContent(verb, "sentence")
		clusterTags = data.getClusterContent(verb, "annotation")
		for i in range(len(clusterSentences)):
			currentTag = ""
			if clusterTags[i] != "U":
				currentTag = clusterTags[i]
			else:
				currentTag = clusterTag

			currentSentence = clusterSentences[i]
			verbObjKey = (actualVerb, getObj(currentSentence ,actualVerb))
			if (verbObjKey[1] != ""):
				if verbObjKey not in verbObjTags.keys():
					verbObjTags[verbObjKey] = []
				verbObjTags[verbObjKey].append(currentTag)
				print("(" + verbObjKey[0] + ", " + verbObjKey[1] + "): " + currentTag)
	return verbObjTags

# Export the tags to a CSV file
def tagsToCSV(tags):
	dictList = []
	verbNouns = tags.keys()
	for vn in verbNouns:
		newDict = {}
		newDict["Verb"] = vn[0]
		newDict["Noun"] = vn[1]
		newDict["Labels"] = tags[vn]
		dictList.append(newDict)
	writeToCSV(dictList, "data/trofi_tags_bis.csv", ["Verb", "Noun", "Labels"])


if __name__ == '__main__':
	# Uncomment to build the TroFi database
	'''
	if BUILD_TROFI:
	
		data1 = ClusteredData.fromFile("data/clustering/verbnet_150_50_200.log-preprocessed", parseVerbNet)
		#print(data1.getClusterContent("3", "words"))
		#
		data2 = ClusteredData.fromFile("data/clustering/200_2000.log-preprocessed", parseNouns)
		#print(data2.getClusterContent("3", "words"))
		#
		data3 = ClusteredData.fromFile("data/clustering/TroFiExampleBase.txt", parseTroFi)
		
		first = FIRST_CLUSTER
		last = LAST_CLUSTER
		if len(sys.argv) >= 3:
			first = int(sys.argv[1])
			last = int(sys.argv[2])
		tags = getVerbObjTags(data3, first, last)
		#tags = getTagsFromCSV("data/trofi_tags_bis.csv")
		print(tags)
		tagsToCSV(tags)
	'''
	buildDB()

