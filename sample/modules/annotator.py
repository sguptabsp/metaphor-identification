# Author : Henri Toussaint
# Latest revision : 01/25/2017

# Annotator file that takes a test string and transforms it into a list of dictionnaries to represent a table of words, lemmas, POS and syntax

import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from .datastructs.annotated_text import AnnotatedText

class Annotator:

	def __init__(self, text=""):
		self.rawText = text
		self.annotatedText = AnnotatedText(self.rawText)

	def addColumn(self, name, annotationFunc):
		newCol = annotationFunc(self.annotatedText)
		self.annotatedText.addColumn(name, newCol)

	def getAnnotatedText(self):
		return self.annotatedText
