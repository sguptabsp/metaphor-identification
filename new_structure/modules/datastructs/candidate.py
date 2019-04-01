from .annotated_text import AnnotatedText

class Candidate:

	def __init__(self, annotatedText, sourceIndex, sourceSpan, targetIndex, targetSpan):
		if (not(isinstance(sourceIndex, int))):
			raise ValueError('Argument sourceIndex not an int.')
		if (not(isinstance(sourceSpan, tuple))):
			raise ValueError('Argument sourceSpan not a tuple.')
		if (not(isinstance(targetIndex, int))):
			raise ValueError('Argument targetIndex not an int.')
		if (not(isinstance(targetSpan, tuple))):
			raise ValueError('Argument targetSpan not a tuple.')
		self.annotatedText = annotatedText
		self.sourceIndex = sourceIndex
		self.sourceSpan = sourceSpan
		self.targetIndex = targetIndex
		self.targetSpan = targetSpan
		# ADD source and target variables?

	def getSource(self):
		if self.annotatedText.isColumnPresent("lemma"):
			return self.annotatedText.getElement(self.sourceIndex, "lemma")
		else:
			return self.annotatedText.getElement(self.sourceIndex, "word")

	def getTarget(self):
		if self.annotatedText.isColumnPresent("lemma"):
			return self.annotatedText.getElement(self.targetIndex, "lemma")
		else:
			return self.annotatedText.getElement(self.targetIndex, "word")			

	def __stringAdder(self, start, end):
		textString = ""
		for i in range(start, end+1):
			if self.annotatedText.isColumnPresent("lemma"):
				textString += self.annotatedText.getElement(i, "lemma")
			else:
				textString += self.annotatedText.getElement(i, "word")
			if (i<end):
				textString+=" "
		return textString

	def getFullSource(self):
		return self.__stringAdder(self.sourceSpan[0], self.sourceSpan[1])

	def getFullTarget(self):
		return self.__stringAdder(self.targetSpan[0], self.targetSpan[1])

	def __str__(self):
		return "Source: " + self.getFullSource() + " || Target: " + self.getFullTarget()
