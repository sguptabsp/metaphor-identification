import csv

class CollocationList:

	def __init__(self):
		self.collocations = {}
		self.size = 0

	def addCollocation(self, adjective, noun, frequency):
		if (not (adjective in self.collocations.keys())):
			currentAdjective = {}
			currentAdjective[noun] = frequency
			self.collocations[adjective] = currentAdjective
			self.size += 1
		else:
			if (not(noun in self.collocations[adjective].keys())):
				self.collocations[adjective][noun] = frequency
			else:
				self.collocations[adjective][noun] += frequency

	def getMostFrequents(self, adjective, n):
		if (adjective in self.collocations.keys()):
			res = sorted(self.collocations[adjective], key=self.collocations[adjective].get, reverse=True)
			size = min(n, len(res))
			return res[:size]
		else:
			print("Source \"" + adjective + "\" not in database.")
			return []

	def getFrequency(self, adjective, noun):
		if (adjective in self.collocations.keys()):
			if (noun in self.collocations[adjective].keys()):
				return self.collocations[adjective][noun]
			else:
				return 0
		else:
			return 0

	def __str__(self):
		s = ""
		for adj in self.collocations.keys():
			for n in self.collocations[adj].keys():
				s += adj + '\t' + n + '\t' + self.collocations[adj][n] + '\n'
		return s

# if adj-noun metaphor: pos_1 = 'jj' and pos_2 = 'nn'
# if verb-noun metaphor: pos_1 = 'v' and pos_2 = 'n'
def parseNgrams(collocations, path, n, pos_1, pos_2):
	with open(path) as tsv:
		for line in csv.reader(tsv, delimiter="\t"):
			# print(line)
			length = len(line)
			for i in range(length-1):
				if (line[i] == pos_1 and line[i + 1].startswith(pos_2)):
					currentAdjIndex = i
					currentNounIndex = i+1
					while (currentNounIndex < len(line) - 1 and line[currentNounIndex + 1].startswith(pos_2)):
						currentNounIndex += 1
		
					noun = line[currentNounIndex-n]
					frequency = int(line[0])
					while (currentAdjIndex >= 0 and line[currentAdjIndex] == pos_1):
						adjective = line[currentAdjIndex-n]
						collocations.addCollocation(adjective, noun, frequency)
						currentAdjIndex -= 1

def parseConcreteness(path):
	concDict = {}
	with open(path) as tsv:
		for line in csv.reader(tsv, delimiter="\t"):
			concDict[line[1]] = int(line[0])
	return concDict