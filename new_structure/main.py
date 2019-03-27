# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

from new_structure.modules.utils import parseCommandLine, getText
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder #Candidate finding functions
from new_structure.modules.darkthoughts import darkthoughtsFunction #Metaphor labeling function
from new_structure.modules.cluster_module import clusteringFunction #Metaphor labeling function
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.datastructs.MetaphorIdentification import MetaphorIdentification


if __name__ == "__main__":

	start_time = time.time()

	args = parseCommandLine()

	#Hash table creation
	metaphorRegistry = Registry()
	metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
	metaphorRegistry.addMLabeler("cluster", clusteringFunction)
	metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
	metaphorRegistry.addCFinder("adjNoun", adjNounFinder)

	#Test if the args are registered 
	if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):

		text = getText(args)
		cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
		mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

		#Object declaration
		object = MetaphorIdentification(text)

		#Annotating the text
		object.annotateText()
		object.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
		object.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
		if args.verbose:
			print(object.getAnnotatedText())

		#Finding candidates
		object.findCandidates(cFinderFunction)
		if args.verbose:
			print(object.getCandidates())

		#labeling Metaphors
		object.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
		if args.verbose:
			print(object.getMetaphors())

		print(object.getMetaphors())

	else:
		print("The candidate finder or the metaphor labeler is incorrect")

	print("--- %s seconds ---" % (time.time() - start_time))