import csv

class LabeledMetaphorList:

	def __init__(self):#, candidateGroup):
		#self.candidateGroup = candidateGroup
		self.results = []
		self.size = 0

	def addResult(self, labeledMetaphor):
		self.results.append(labeledMetaphor)
		self.size += 1

	def getResult(self, index):
		return self.results[index]

	def __iter__(self):
		return iter(self.results)

	def writeToCSV(self, path):
		dicList = []
		columns = ["source", "target", "result", "confidence"]
		for r in self.results:
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
		resultsString = ""
		for i in range(self.size):
			resultsString += str(self.results[i]) + "\n"
		return resultsString