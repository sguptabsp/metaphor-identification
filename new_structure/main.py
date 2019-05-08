# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

# Metaphor labeling functions
from new_structure.modules.cluster_module import clusteringFunction
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.kmeans_abs_ratings_cosine_edit_distance import identify_metaphors_abstractness_cosine_edit_dist
# Data structures
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
# Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
# Utils functions
from new_structure.modules.utils import parseCommandLine, getText
# MetaphorIdentification objects
from new_structure.modules.datastructs.metaphor_identification_list import MetaphorIdentificationList


if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Hash table creation
    metaphorRegistry = Registry()
    metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
    metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
    metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
    metaphorRegistry.addMLabeler("cluster", clusteringFunction)
    metaphorRegistry.addMLabeler("kmeans", identify_metaphors_abstractness_cosine_edit_dist)


    #Test if the args are registered in the hashtable
    if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):

        text = getText(args)
        cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
        mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

        #Object declaration
        list = MetaphorIdentificationList(text)

        #Step 1: Annotating the text
        list.annotateText()
        list.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        list.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if args.verbose:
            print(list.getAnnotatedText())

        #Step 2: Finding candidates
        list.findCandidates(cFinderFunction)
        if args.verbose:
            print(list.getCandidates())

        #Step 3: Labeling Metaphors
        list.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
        if args.verbose:
            print(list.getMetaphors())

        print(list.getMetaphors())

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

    print("--- %s seconds ---" % (time.time() - start_time))


