from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import csv

class AnnotatedText:

	def __init__(self, text=""):
		self.rawText = text
		self.table = []
		self.words = word_tokenize(self.rawText)
		self.size = len(self.words)
		self.columns = ["index", "word"]
		for i in range(self.size):
			currentWord = {}
			currentWord["index"] = i
			currentWord["word"] = self.words[i]
			self.table.append(currentWord)

	def getText(self):
		return self.rawText

	def getLine(self, index):
		return self.table[index]

	def getColumn(self, columnName):
		column = []
		for i in range(self.size):
			column.append(self.table[i][columnName])
		return column

	def getElement(self, index, columnName):
		return self.table[index][columnName]

	def isColumnPresent(self, columnName):
		return columnName in self.columns

	def addColumn(self, name, column):
		if(name in self.columns):
			return

		self.columns.append(name)
		for i in range(self.size):
			self.table[i][name] = column[i]

	def consistencyCheck(self):
		for i in range(self.size):
			for c in self.columns:
				if (list(self.table[i].keys()).count(c)!=1):
					return False
		return True

	def writeToCSV(self, path):
		with open(path, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, self.columns)
			writer.writeheader()
			writer.writerows(self.table)

	def __str__(self):
		tableString = ""
		for i in range(self.size):
			tableString += str(self.table[i]) + "\n"
		return tableString