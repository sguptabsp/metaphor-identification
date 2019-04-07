# Author : Thomas Buffagni
# Latest revision : 03/26/2019

import time

from new_structure.modules.cluster_module import clusteringFunction
# Metaphor labeling functions
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.datastructs.metaphor_abstractness_identification import MetaphorAbstractnessIdentification
from new_structure.modules.datastructs.metaphor_identification import MetaphorIdentification
# Data structures
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.kmeans_abs_ratings_cosine_edit_distance import \
    identify_metaphors_abstractness_cosine_edit_dist
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
# Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
from new_structure.modules.utils import parseCommandLine, getText

if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Hash table creation
    metaphorRegistry = Registry()
    # metaphorRegistry.addMLabeler("kmeans_abs_ratings", kmeans_abs_ratings)
    # metaphorRegistry.addMLabeler("kmeans_abs_ratings_cosine_similarity", kmeans_abs_ratings_cosine_similarity)
    metaphorRegistry.addMLabeler("kmeans_abs_ratings_cosine_edit_distance",
                                 identify_metaphors_abstractness_cosine_edit_dist)
    metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
    metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
    metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
    metaphorRegistry.addMLabeler("cluster", clusteringFunction)

    # Test if the args are registered in the hashtable
    if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):

        text = getText(args)
        cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
        mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

        # Object declaration
        if "kmeans_abs_ratings" in args.mlabeler:
            object = MetaphorAbstractnessIdentification(args.file)

        else:

            object = MetaphorIdentification(text)

        # Step 1: Annotating the text
        object.annotateText()
        object.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        object.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if args.verbose:
            print(object.getAnnotatedText())

        # Step 2: Finding candidates
        object.findCandidates(cFinderFunction)
        if args.verbose:
            print(object.getCandidates())

        # Step 3: Labeling Metaphors
        object.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
        if args.verbose:
            print(object.getMetaphors())

        print(object.getMetaphors())

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

    print("--- %s seconds ---" % (time.time() - start_time))

    print("Raw Text")
    print(object.rawText)
    print("\n")
    print("Annotated Text")
    print("size:", object.annotatedText.size)
    print("words:", object.annotatedText.words)
    print("columns:", object.annotatedText.columns)
    print("table:", object.annotatedText.table)
    print("\n")
    print("Candidates: class candidateGroup")
    print("size:", object.candidates.size)
    for c in object.candidates.candidates:
        print("sourceIndex:", c.sourceIndex)
        print("sourceSpan:", c.sourceSpan)
        print("targetIndex:", c.targetIndex)
        print("targetSpan:", c.targetSpan)
    print("\n")
    print("Metaphors: class MetaphorGroup")
    print("size:", object.metaphors.size)
    for m in object.metaphors.metaphors:
        print("candidate:", m.candidate)
        print("result:", m.result)
        print("confidence:", m.confidence)
