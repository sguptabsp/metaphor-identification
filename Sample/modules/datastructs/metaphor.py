class Metaphor:

	def __init__(self, candidate, result, confidence):
		if (not(isinstance(result, bool))):
			raise ValueError('Argument result not a boolean.')
		if (not(isinstance(confidence, float))):
			raise ValueError('Argument confidence not a float.')
		self.candidate = candidate
		self.predictedLabel = result
		self.confidence = confidence

	def getSource(self):
		return self.candidate.getFullSource()

	def getTarget(self):
		return self.candidate.getFullTarget()

	def getPredictedLabel(self):
		return self.predictedLabel

	def getConfidence(self):
		return self.confidence

	def getLabel(self):
		return self.candidate.getLabel()

	def __str__(self):
		source = self.getSource()
		target = self.getTarget()
		return "Source: " + source + " || Target: " + target + " || Result: " + str(self.predictedLabel) + " || Confidence: " + str(self.confidence)