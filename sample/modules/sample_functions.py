from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from .datastructs.annotated_text import AnnotatedText
from .datastructs.candidate_group import CandidateGroup
from .datastructs.metaphor_candidate import MetaphorCandidate
from .datastructs.labeled_metaphor_list import LabeledMetaphorList
from .datastructs.labeled_metaphor import LabeledMetaphor
from .registry import metaphorRegistry

def getWordnetPos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def posFunction(annotatedText):
	finalPos = []
	sentence = []
	if (annotatedText.isColumnPresent("word")):
		sentence = annotatedText.getColumn("word")
	else:
		return finalPos
	pos = pos_tag(sentence)
	for i in range(len(pos)):
		finalPos.append(pos[i][1])
	return finalPos


def lemmatizingFunction(annotatedText):
	lemm = WordNetLemmatizer()
	posTags = []
	finalLem = []
	sentence = []
	if (annotatedText.isColumnPresent("word")):
		sentence = annotatedText.getColumn("word")
	else:
		return finalPos
	if (annotatedText.isColumnPresent("POS")):
		posTags = annotatedText.getColumn("POS")
	else:
		return finalLem

	for i in range(len(sentence)):
		curentTag = getWordnetPos(posTags[i])
		if (curentTag):
			finalLem.append(lemm.lemmatize(sentence[i], curentTag))
		else:
			finalLem.append(lemm.lemmatize(sentence[i]))
	return finalLem


def testIDFunction(annotatedText):
	candidates = CandidateGroup()
	testCandidate = MetaphorCandidate(annotatedText, 2, (0, 2), 6, (6,6))
	candidates.addCandidate(testCandidate)
	return candidates

def adjNounFinder(annotatedText):
	candidates = CandidateGroup()
	POScolumn = annotatedText.getColumn("POS")
	candidate = []
	currentAdjectives = []
	for i in range(len(POScolumn)-1):
		if (POScolumn[i] == 'JJ' and POScolumn[i+1].startswith('NN')):
			currentAdjIndex = i
			currentNounIndex = i+1
			while (currentNounIndex < len(POScolumn) and POScolumn[currentNounIndex].startswith('NN')):
				currentNounIndex += 1
			while (currentAdjIndex >= 0 and POScolumn[currentAdjIndex] == 'JJ'):
				newCandidate = MetaphorCandidate(annotatedText, currentAdjIndex, (currentAdjIndex, currentAdjIndex), currentNounIndex-1, (i+1,currentNounIndex-1))
				candidates.addCandidate(newCandidate)
				currentAdjIndex -= 1

	return candidates

metaphorRegistry.addCFinder("adjNoun", adjNounFinder)

# Finds the verb and the next noun in the sentence
def verbNounFinder(annotatedText):
	candidates = CandidateGroup()
	POScolumn = annotatedText.getColumn("POS")
	wordColumn = annotatedText.getColumn("word")
	candidate = []
	currentAdjectives = []
	for i in range(len(POScolumn)-1):
		if POScolumn[i].startswith('VB'):
			currentVerbIndex = i
			currentNounIndex = i
			while (currentNounIndex < len(POScolumn) and wordColumn[currentNounIndex]!="." and not(POScolumn[currentNounIndex].startswith('NN'))):
				currentNounIndex += 1
			if currentNounIndex < len(POScolumn) and POScolumn[currentNounIndex].startswith('NN'):
				newCandidate = MetaphorCandidate(annotatedText, currentVerbIndex, (currentVerbIndex, currentVerbIndex), currentNounIndex, (currentNounIndex, currentNounIndex))
				candidates.addCandidate(newCandidate)

	return candidates

metaphorRegistry.addCFinder("verbNoun", verbNounFinder)


# TODO: Write function that finds a verb and its object using a dependancy parser
'''
def verbObjFinder(annotatedText):
	candidates = CandidateGroup()
	text = annotatedText.getText()

	parser = StanfordDependencyParser()
	lemmatizer = WordNetLemmatizer()
	dependency_tree = [list(line.triples()) for line in parser.raw_parse(text)]
	dependencies = dependency_tree[0]
	verbLemma = ""
	obj = ""
	currentIndex = 0
	for dep in dependencies:
		if annotatedText.getElement(currentIndex, "word") in [',', ';', '-', '.', '?', '!']:
			currentIndex += 1

		if  "VB" in dep[0][1]:
			verbLemma = lemmatizer.lemmatize(dep[0][0], wordnet.VERB)
			verbIndex = currentIndex
			if ("obj" in dep[1] or "nsubjpass" in dep[1]):
				obj = dep[2][0]
				# NEED TO BE ABLE TO CREATE CANDIDATE FROM WORDS INSTEAD OF INDEXES
				#newCandidate = MetaphorCandidate(annotatedText, objIndex, (objIndex, objIndex), )
		currentIndex += 1
'''

def testLabelFunction(candidates):
	results = LabeledMetaphorList()
	for c in candidates:
		if (c.getSource()[0] == c.getTarget()[0]):
			results.addResult(LabeledMetaphor(c, True, 0.5))
		else:
			results.addResult(LabeledMetaphor(c, False, 0.5))
	
	return results