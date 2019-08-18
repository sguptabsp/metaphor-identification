class Cluster:

	def __init__(self, entry="", content={}, contentEntries=[]):
		self.entry = entry # title/index/entry of the cluster
		self.content = content # content of the cluster (sentence, lists of words, words, annotation ...)
		self.contentEntries = contentEntries # keys of the content ("sentence", "words", "annotation", ...)

	def getEntry(self):
		return self.entry

	def getContent(self, contentEntry):
		if contentEntry in self.contentEntries:
			return self.content[contentEntry]
		else:
			return None

	def isInCluster(self, contentEntry, contentValue):
		return contentValue in self.content[contentEntry]

	def getContentEntries(self):
		return self.contentEntries