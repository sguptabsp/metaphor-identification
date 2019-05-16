# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time


# Metaphor labeling functions
from new_structure.modules.cluster_module import clusteringFunction
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.kmeans_abs_ratings_cosine_edit_distance import identify_metaphors_abstractness_cosine_edit_dist
# Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
from new_structure.modules.utils import parseCommandLine, getText
# Data structures
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
from new_structure.modules.datastructs.metaphor_identification_list import MetaphorIdentificationList


if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Hash table creation
    metaphorRegistry = Registry()
    metaphorRegistry.addMLabeler("kmeans",identify_metaphors_abstractness_cosine_edit_dist)
    metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
    metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
    metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
    metaphorRegistry.addMLabeler("cluster", clusteringFunction)

    #Test if the args are registered in the hashtable
    if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):

        texts, sources, targets = getText(args)
        cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
        mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

        #Object declaration
        list = MetaphorIdentificationList(texts)

        #Step 1: Annotating the text
        list.annotateAllTexts()
        list.allAnnotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        list.allAnnotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if args.verbose:
            print(list.getAnnotatedText())

        #Step 2: Finding candidates
        if not args.cgenerator:
            list.findAllCandidates(cFinderFunction)
            if args.verbose:
                print(list.getCandidates())
        else:
            list.allCandidatesFromColumns(sources, targets, [i for i in range(len(sources))])

        #Step 3: Labeling Metaphors
        list.labelAllMetaphors(mLabelerFunction, args.cfinder, verbose=args.verbose)
        if args.verbose:
            print(list.getAllMetaphors())

        print(list.getAllMetaphors())

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

    print("--- %s seconds ---" % (time.time() - start_time))
