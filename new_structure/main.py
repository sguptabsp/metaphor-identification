# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

from new_structure.modules.utils import parseCommandLine, getText
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
#Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
#Metaphor labeling functions
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.cluster_module import clusteringFunction
#Data structures
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.datastructs.MetaphorIdentification import MetaphorIdentification


if __name__ == "__main__":

	start_time = time.time()

	args = parseCommandLine()

	#Hash table creation
	metaphorRegistry = Registry()
	metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
	metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
	metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
	metaphorRegistry.addMLabeler("cluster", clusteringFunction)

	#Test if the args are registered in the hashtable
	if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):

		text = getText(args)
		cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
		mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

		#Object declaration
		object = MetaphorIdentification(text)

		#Step 1: Annotating the text
		object.annotateText()
		object.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
		object.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
		if args.verbose:
			print(object.getAnnotatedText())

		#Step 2: Finding candidates
		object.findCandidates(cFinderFunction)
		if args.verbose:
			print(object.getCandidates())

		#Step 3: Labeling Metaphors
		object.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
		if args.verbose:
			print(object.getMetaphors())

		print(object.getMetaphors())

	else:
		print("The candidate finder or the metaphor labeler is incorrect")

	print("--- %s seconds ---" % (time.time() - start_time))