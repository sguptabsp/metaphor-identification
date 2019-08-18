# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

# Metaphor labeling functions
from new_structure.modules.cluster_module import clusteringFunction
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.datastructs.metaphor_identification import MetaphorIdentification
from new_structure.modules.kmeans_abs_ratings_cosine_edit_distance import \
    identify_metaphors_abstractness_cosine_edit_dist
# Data structures
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
# Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
from new_structure.modules.utils import parseCommandLine, getText

if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Initilization
    MI = MetaphorIdentification()

    # Registering the Candidate Finders and the Metaphor Labelers
    MI.addCFinder("verbNoun", verbNounFinder)
    MI.addCFinder("adjNoun", adjNounFinder)
    MI.addMLabeler("darkthoughts", darkthoughtsFunction)
    MI.addMLabeler("cluster", clusteringFunction)
    MI.addMLabeler("kmeans", identify_metaphors_abstractness_cosine_edit_dist)

    # Test if the args are registered in the hashtable
    if MI.isMLabeler(args.mlabeler) and MI.isCFinder(args.cfinder):

        texts, sources, targets = getText(args)
        cFinderFunction = MI.getCFinder(args.cfinder)
        mLabelerFunction = MI.getMLabeler(args.mlabeler)

        # Loading the texts in the Metaphor Identification Object
        MI.addText(texts)

        # Step 1: Annotating the text
        MI.annotateAllTexts()
        MI.allAnnotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        MI.allAnnotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if args.verbose:
            print(MI.getAnnotatedText())

        # Step 2: Finding candidates
        if not args.cgenerator:
            MI.findAllCandidates(cFinderFunction)
            if args.verbose:
                print(MI.getCandidates())
        else:
            MI.allCandidatesFromColumns(sources, targets, [i for i in range(len(sources))])

        # Step 3: Labeling Metaphors
        MI.labelAllMetaphors(mLabelerFunction, args.cfinder, verbose=args.verbose)
        if args.verbose:
            print(MI.getAllMetaphors())

        print(MI.getAllMetaphors())

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

    print("--- %s seconds ---" % (time.time() - start_time))
