from .datastructs.cluster import Cluster
import csv
import re

def parseVerbNet(file, data):
	with open(file, 'r') as file:
		lines = file.readlines()

		data.contentEntries.append("words")
		for l in lines:
			if l.startswith(" <"):
				data.clusterEntries.append(re.findall(r'"([^"]*)"', l)[0])
			elif l.startswith("-"):
				continue
			else:
				newClusterContent = {}
				newClusterContent["words"] = []

				for word in l.split():
					newClusterContent["words"].append(word)

				newCluster = Cluster(data.clusterEntries[-1], newClusterContent, data.contentEntries)
				data.clusters.append(newCluster)

def parseNouns(file, data):
	with open(file, 'r') as file:
		lines = file.readlines()

		data.contentEntries.append("words")
		for l in lines:
			newClusterContent = {}
			newClusterContent["words"] = []

			wordsInLine = l.split()
			data.clusterEntries.append(wordsInLine[0][7:])

			for word in wordsInLine[1:]:
				newClusterContent["words"].append(word)

			newCluster = Cluster(data.clusterEntries[-1], newClusterContent, data.contentEntries)
			data.clusters.append(newCluster)

def parseTroFi(file, data):
	with open(file, 'r') as file:
		lines = file.readlines()

		data.contentEntries.append("index")
		data.contentEntries.append("reference")
		data.contentEntries.append("annotation")
		data.contentEntries.append("sentence")

		currentWord = ""
		newClusterContent = {}
		intex = 0
		for l in lines:
			if l == "********************\n" or l == '\n':
				newCluster = Cluster(data.clusterEntries[-1], newClusterContent, data.contentEntries)
				data.clusters.append(newCluster)
			elif l.startswith("***"):
				currentWord = l[3:-4]
			elif l == "*nonliteral cluster*\n":
				data.clusterEntries.append(currentWord+"-NL")
				newClusterContent = {}
				index = 0
				newClusterContent['index'] = []
				newClusterContent["reference"] = []
				newClusterContent["annotation"] = []
				newClusterContent["sentence"] = []
			elif l == "*literal cluster*\n":
				data.clusterEntries.append(currentWord+"-L")
				newClusterContent = {}
				index = 0
				newClusterContent['index'] = []
				newClusterContent["reference"] = []
				newClusterContent["annotation"] = []
				newClusterContent["sentence"] = []
			else:
				elements = l.split('\t')
				newClusterContent["reference"].append(elements[0])
				newClusterContent["annotation"].append(elements[1])
				newClusterContent["sentence"].append(elements[2][:-4])
				newClusterContent["index"].append(index)
				index += 1

