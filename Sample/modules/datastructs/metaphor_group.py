import csv

class MetaphorGroup:

	def __init__(self):#, candidateGroup):
		#self.candidateGroup = candidateGroup
		self.metaphors = []
		self.size = 0

	def addMetaphor(self, metaphor):
		self.metaphors.append(metaphor)
		self.size += 1

	def getMetaphor(self, index):
		return self.metaphors[index]

	def getSize(self):
		return self.size

	def getResults(self):
		results = list()
		for i in range(self.size):
			results.append(self.metaphors[i].getResult())
		return results

	def getConfidences(self):
		confidences = list()
		for i in range(self.size):
			confidences.append(self.metaphors[i].getConfidence())
		return confidences

	def filterByConfidence(self, min_confidence):
		mg = MetaphorGroup()
		for m in self.metaphors:
			if m.getConfidence() >= min_confidence:
				mg.addMetaphor(m)
		return mg

	def __iter__(self):
		return iter(self.metaphors)

	def writeToCSV(self, path):
		dicList = []
		columns = ["source", "target", "result", "confidence"]
		for r in self.metaphors:
			currentResult = {}
			currentResult["source"] = r.getSource()
			currentResult["target"] = r.getTarget()
			currentResult["result"] = r.getResult()
			currentResult["confidence"] = r.getConfidence()
			dicList.append(currentResult)

		with open(path, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, columns)
			writer.writeheader()
			writer.writerows(dicList)

	def __str__(self):
		metaphorsString = ""
		for i in range(self.size):
			metaphorsString += str(self.metaphors[i]) + "\n"
		return metaphorsString