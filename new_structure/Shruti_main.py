import pandas as pd
import time
from new_structure.modules.utils import parseCommandLine
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder  # Candidate finding functions
from new_structure.modules.darkthoughts import darkthoughtsFunction  # Metaphor labeling function
from new_structure.modules.cluster_module import clusteringFunction  # Metaphor labeling function
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.datastructs.metaphor_identification import MetaphorIdentification


if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Hash table creation
    metaphorRegistry = Registry()
    metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
    metaphorRegistry.addMLabeler("cluster", clusteringFunction)
    metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
    metaphorRegistry.addCFinder("adjNoun", adjNounFinder)

    # Test if the args are registered in the hashtable
    if metaphorRegistry.isMLabeler(args.mlabeler) and metaphorRegistry.isCFinder(args.cfinder):
        cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
        mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)

        # Extracting data from xlsx files
        file_path = '../../Shruti/Datasets_ACL2014.xlsx'
        MET_AN_EN_TEST = pd.read_excel(file_path, sheet_name='MET_AN_EN', usecols=[5, 6, 7])
        MET_AN_EN_TEST['class'] = 1
        LIT_AN_EN_TEST = pd.read_excel(file_path, sheet_name='LIT_AN_EN', usecols=[5, 6, 7])
        LIT_AN_EN_TEST['class'] = 0

        data = pd.concat([LIT_AN_EN_TEST, MET_AN_EN_TEST])
        data = pd.DataFrame(data)

        # Identifying the metaphors
        metaphor_list = []
        for i in range(10):#(data.shape[0]):
            text = data.iloc[i]["sentence"]
            # Object declaration
            object = MetaphorIdentification(text)

            # Annotating the text
            object.annotateText()
            object.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
            object.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
            if args.verbose:
                print(object.getAnnotatedText())

            # Finding candidates
            object.findCandidates(cFinderFunction)
            if args.verbose:
                print(object.getCandidates())

            print(object.annotatedText)
            for c in object.candidates:
                print(c.getSource())
                print(c.getTarget())

            # labeling Metaphors
            # object.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
            # if args.verbose:
            #     print(object.getMetaphors())

            # Find a type to each labeled metaphor (TP, TN, FP, FN)
            # for met in object.getMetaphors():
            #     metaphor = {'text': text, 'target': met.getTarget(), 'source': met.getSource(), 'result': met.getResult(), 'confidence': met.getConfidence()}
            #     metaphor_list.append(metaphor)

            # Progess of the loop
            print(i, "out of", data.shape[0])

        metaphor_df = pd.DataFrame(metaphor_list)
        write_csv_path = "./data/ShrutiMetaphors.csv"
        metaphor_df.to_csv(path_or_buf=write_csv_path, index=False, encoding='utf-8')

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

print("--- %s seconds ---" % (time.time() - start_time))

