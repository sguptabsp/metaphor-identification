class Categories:

	def __init__(self):
		self.categories = {}
		self.words = {}

	def addWord(self, category, word):
		if not category in self.categories.keys():
			self.categories[category] = []
		self.categories[category].append(word)
		if not word in self.words.keys():
			self.words[word] = []
		self.words[word].append(category)

	def getCategoriesFromWord(self, word):
		return self.words[word]

	def getCategories(self):
		return self.categories.keys()

	def getWordsFromCategory(self, category):
		return self.categories[category]

	def getCategoriesFromWords(self, n, words):
		res = []
		for c in self.categories.keys():
			currentCategory = self.categories[c]
			if (len(set(currentCategory).intersection(words)) >= n):
				res.append(c)
		return res

	def isWordInCat(self, word, category):
		if (word in self.words.keys()):
			return category in self.words[word]
		else:
			return False

	def __str__(self):
		return str(self.categories)

def parseCategories(path):
	categories = Categories()
	currentCategory = ""
	with open(path) as file:
		for line in file:
			if not(line.startswith('\t')):
				currentCategory = line.partition('.')[-1].rpartition('\n')[0]
			else:
				word = line.partition('\t')[-1].rpartition(' ')[0]
				categories.addWord(currentCategory, word)
	return categories