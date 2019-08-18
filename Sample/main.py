# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

# Metaphor labeling functions
from Sample.modules.cluster_module import clusteringFunction, clusteringFunction_2
from Sample.modules.darkthoughts import darkthoughtsFunction, darkthoughtsFunction_2
from Sample.modules.new_cluster_module import newClusterModule
from Sample.modules.kmeans_abs_ratings_cosine_edit_distance import identify_metaphors_abstractness_cosine_edit_dist
# Candidate finding functions
from Sample.modules.sample_functions import verbNounFinder, adjNounFinder
from Sample.modules.utils import parseCommandLine, getText
# Data structures
from Sample.modules.sample_functions import posFunction, lemmatizingFunction
from Sample.modules.datastructs.metaphor_identification import MetaphorIdentification


if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    #Initilization
    MI = MetaphorIdentification()

    # Registering the Candidate Finders and the Metaphor Labelers
    MI.addCFinder("verbNoun", verbNounFinder)
    MI.addCFinder("adjNoun", adjNounFinder)
    MI.addMLabeler("darkthoughts", darkthoughtsFunction_2)
    MI.addMLabeler("cluster", clusteringFunction_2)
    MI.addMLabeler('newCluster', newClusterModule)
    MI.addMLabeler("kmeans", identify_metaphors_abstractness_cosine_edit_dist)

    #Test if the args are registered in the hashtable
    if MI.isMLabeler(args.mlabeler) and MI.isCFinder(args.cfinder):

        texts, sources, targets, labels = getText(args)

        # Loading the texts in the Metaphor Identification Object
        MI.addText(texts)

        #Step 1: Annotating the text
        MI.annotateAllTexts()
        MI.allAnnotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        MI.allAnnotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if args.verbose:
            print(MI.getAnnotatedText())

        #Step 2: Finding candidates
        if not args.cgenerator:
            MI.findAllCandidates(args.cfinder) # Call the candidate finder specified in the command line by the user
            if args.verbose:
                print(MI.getCandidates())
            candidatesID = args.cfinder
        else:
            MI.allCandidatesFromFile(sources, targets, labels, [i for i in range(len(sources))])
            candidatesID = 'fromFile'

        #Step 3: Labeling Metaphors
        cand_type = args.cfinder
        MI.labelAllMetaphorsOneGroup(args.mlabeler, candidatesID, cand_type, verbose=args.verbose)  # Call the metaphor labeler specified in the command line by the user
        if args.verbose:
            print(MI.getAllMetaphors())

        if args.csv:
            MI.resultsToCSV(args.csv)


    else:
        print("The candidate finder or the metaphor labeler is incorrect")

    print("--- %s seconds ---" % (time.time() - start_time))
