from .candidate import Candidate

class CandidateGroup:

	def __init__(self):
		self.candidates = []
		self.size = 0

	def addCandidate(self, candidate):
		self.candidates.append(candidate)
		self.size += 1

	def getCandidate(self, index):
		return self.candidates[index]

	def __iter__(self):
		return iter(self.candidates)

	def __str__(self):
		candidatesString = ""
		for i in range(self.size):
			candidatesString += str(self.candidates[i]) + "\n"
		return candidatesString
