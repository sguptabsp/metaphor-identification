class WordClusterList:

	def __init__(self):
		self.clusters = []
		self.wordList = {} #dictionnary of clusters, to quickly find the corresponding cluster to a given word
		self.fullClusterDic = {} #dictionnary of word lists, to quickly find the full list of words in a cluster

	@classmethod
	def fromFile(cls, file, parsingFunction):
		data = cls()
		parsingFunction(file, data)
		return data

	def addCluster(self, name, content=[]):
		self.clusters.append(name)
		self.fullClusterDic[name] = []
		#self.doneClusterDic[name] = []
		for word in content:
			self.fullClusterDic[name].append(word)
			self.wordList[word] = name

	def addWord(self, word, cluster):
		self.fullClusterDic[cluster].append(word)
		self.wordList[word] = cluster

	def getCluster(self, word):
		return self.wordList[word]

	def getWords(self, cluster):
		return self.fullClusterDic[cluster]

	def isWordInCluster(self, word):
		return word in self.wordList.keys()

	def __iter__(self):
		return iter([self.fullClusterDic[c] for c in self.clusters])
	
	def __str__(self):
		s = ""
		for c in self.clusters:
			s += "Cluster " + c + ": " + str(self.fullClusterDic[c]) + "\n"
		return s