import csv

from . import cluster

class ClusteredData:

	def __init__(self):
		self.clusters = [] # list of clusters
		self.clusterEntries = [] # list of cluster entries
		self.contentEntries = [] # list of cluster content entries

	def getCluster(self, entry):
		for c in self.clusters:
			if c.getEntry() == entry:
				return c
		return None

	def getClusterContent(self, clusterEntry, contentEntry):
		if contentEntry not in self.contentEntries:
			return None

		for c in self.clusters:
			if c.getEntry() == clusterEntry:
				return c.getContent(contentEntry) 
	
	def getEntries(self):
		return self.clusterEntries

	def getCluster(self, contentEntry, contentValue):
		for c in self.clusters:
			if c.isInCluster(contentEntry, contentValue):
				return c.getEntry()
		return None

	def __iter__(self):
		return iter(self.clusters)

	@classmethod
	def fromFile(cls, file, parsingFunction):
		data = cls()
		parsingFunction(file, data)
		return data

