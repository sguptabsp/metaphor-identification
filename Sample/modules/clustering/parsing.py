#from . import WordClusterList
import re

def parseVerbNet(file, data):
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
				data.addCluster(name, content)

def parseNouns(file, data):
	with open(file, 'r') as file:
		lines = file.readlines()
		name = ""
		content = []

		for l in lines:	
			newClusterContent = {}
			newClusterContent["words"] = []

			wordsInLine = l.split()
			name = wordsInLine[0][7:]
			content = wordsInLine[1:]

			data.addCluster(name, content)