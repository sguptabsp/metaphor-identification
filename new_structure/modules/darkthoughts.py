from nltk.corpus import wordnet
from .datastructs.annotated_text import AnnotatedText
from .datastructs.candidate_group import CandidateGroup
from .datastructs.candidate import Candidate
from .datastructs.metaphor_group import MetaphorGroup
from .datastructs.metaphor import Metaphor
from .datastructs.ngrams import CollocationList, parseConcreteness, parseNgrams
from .datastructs.categories import Categories, parseCategories

ADJ_FILES = [ ("data/an2.txt", 2), ("data/an3.txt", 3), ("data/an4.txt", 4), ("data/an5.txt", 5)]
VERB_FILES = [("data/vn2.txt", 2)] # This file is a filtered version of vn2_v1: it only keep the lines with frequency > 1000
NGRAMS_FILES = {"adjNoun": ADJ_FILES, "verbNoun": VERB_FILES}

POS = [{"adjNoun": "jj", "verbNoun": "v"}, {"adjNoun": "nn", "verbNoun": "n"}]

CONCRETENESS_FILE = "data/concreteness.txt"
CAT_FILE = "data/categories.txt"
TOP_SIZE = 60
K = 30
T = 10

def darkthoughtsFunction(candidates, cand_type, verbose):
	# INITIALIZATIONS OF CLASSES AND DATA STRUCTURES
	results = MetaphorGroup()
	collocations = CollocationList()

	for f in NGRAMS_FILES[cand_type]:
		parseNgrams(collocations, f[0], f[1], POS[0][cand_type], POS[1][cand_type])

	concrDict = parseConcreteness(CONCRETENESS_FILE)
	categories = parseCategories(CAT_FILE)

	for c in candidates:
		source = c.getSource()
		target = c.getTarget()
		result = True
		confidence = 1.0
		if verbose:
			print("###############################################################################")
			print("SOURCE: " + source)
			print("TARGET: " + target)

		# PART 1 OF ALGORITHM: GET THE NOUNS MOST FREQUENTLY MODIFIED BY SOURCE
		topModified = collocations.getMostFrequents(source, TOP_SIZE)
		if verbose:
			print("NOUNS MOST FREQUENTLY MODIFIED BY SOURCE")
			print(topModified)
			print()

		# PART 2: GET THE K MOST CONCRETE NOUNS IN TOPMODIFIED
		topConcrete = sorted(topModified, key=lambda x: concrDict.get(x, 0), reverse=True)[:K]
		if verbose:
			print(str(K) + " MOST CONCRETE NOUNS IN THE PREVIOUS LIST")
			print(topConcrete)
			print()

		# PART 3: GET THE SEMANTIC CATEGORIES CONTAINING AT LEAST T NOUNS IN TOPCONCRETE
		topCategories = categories.getCategoriesFromWords(T, topConcrete)
		if verbose:
			print("CATEGORIES CONTAINING AT LEAST " + str(T) + " NOUNS FROM THE PREVIOUS LIST")
			print(topCategories)
			print()

		# PART 4: IF THE NOUN BELONGS TO ONE OF THE CATEGORIES IN TOPCATEGORIES, THEN IT IS LITERAL
		if verbose:
			print("CATEGORIES OF THE TARGET")
			print(categories.getCategoriesFromWord(target))
			print()


		###### VERSION WITHOUT CONFIDENCE ######
		#for cat in topCategories:
		#	if categories.isWordInCat(target, cat):
		#		result = False
		#		break
		########################################



		###### VERSION WITH CONFIDENCE #####
		coef = 0.0
		intersection = 0.0
		ratio = 0.0
		if (topCategories):
			intersection = len(set(topCategories).intersection(categories.getCategoriesFromWord(target)))
			ratio = intersection/len(topCategories)
			if (ratio == 1.0):
				coef = 1.0
			else:
				coef = 1-0.1/(0.1+ratio)

		if (coef > 0.5):
			result = False
			confidence = coef
		else:
			confidence = 1-coef

		if verbose:
			print("LENGTH OF THE INTERSECTION BETWEEN THE LIST OF CATEGORIES OF THE WORD AND THE LIST OF CATEGORIES CONTAINING THE MOST CONCRETE NOUNS: " + str(intersection))
			print("RATIO INTERSECTION/CONCRETE_CATEGORIES: " + str(ratio))
			print("CONFIDENCE: " + str(confidence))
			print()

		#####################################

		results.addMetaphor(Metaphor(c, result, confidence))

	return results
