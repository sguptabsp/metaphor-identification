import re
import pandas as pd
import time
from new_structure.modules.utils import parseCommandLine
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder  # Candidate finding functions
from new_structure.modules.darkthoughts import darkthoughtsFunction  # Metaphor labeling function
from new_structure.modules.cluster_module import clusteringFunction  # Metaphor labeling function
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.datastructs.metaphor_identification import MetaphorIdentification
from new_structure.modules.TroFi.TroFiParsing import parse_file


# met is a Metaphor object (modules/datastructs/metaphor.py)
# data is Series (pandas library) object with columns: verb, cluster, id, status, text (modules/TroFi/TrofiParsing.py)
def TroFiVerification(met, data):
    cluster = data["cluster"]
    labeling_result = met.getResult()

    # If the metaphor is not on the correct word in the sentence (unlabeled data)
    if met.getSource().lower() != data["verb"].lower():
        return "unknown"

    # If the metaphor is on the correct word
    else:
        if labeling_result == True:
            if cluster == "nonliteral":
                type = "true positive"
            elif cluster == "literal":
                type = "false positive"

        else:
            if cluster == "nonliteral":
                type = "false negative"
            elif cluster == "literal":
                type = "true negative"

    return type


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

        # Extracting data from text file
        textfile_path = "./data/TroFiDataset.txt"
        regex_dict = {
            'verb': re.compile(r'\*{3}[a-z]+\*{3}'),
            'cluster': re.compile(r'\*(non)?literal cluster\*'),
            'id': re.compile(r'wsj[0-9]{2}:[0-9]+\s'),
            'status': re.compile(r'[UNL]\W'),
            'text': re.compile(r'[A-Z][^\s].+\.\/\.')
        }

        data = parse_file(textfile_path, regex_dict)
        data.to_csv("./data/TroFiDataset.csv", index=False)

        verbs = set(data["verb"].tolist())

        # Identifying the metaphors
        metaphor_list = []
        for i in range(10):#(data.shape[0]):
            text = data.iloc[i]["text"]
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

            # labeling Metaphors
            object.labelMetaphors(mLabelerFunction, args.cfinder, args.verbose)
            if args.verbose:
                print(object.getMetaphors())

            # Find a type to each labeled metaphor (TP, TN, FP, FN)
            for met in object.getMetaphors():
                type = TroFiVerification(met, data.iloc[i])
                metaphor = {'text': text, 'target': met.getTarget(), 'source': met.getSource(), 'result': met.getResult(), 'confidence': met.getConfidence(), 'type': type}
                metaphor_list.append(metaphor)

            # Progess of the loop
            print(i, "out of", data.shape[0])

        metaphor_df = pd.DataFrame(metaphor_list)
        write_csv_path = "./data/TroFiMetaphors.csv"
        metaphor_df.to_csv(path_or_buf=write_csv_path, index=False)

        # Displaying the results
        count_false_positive = metaphor_df[metaphor_df.type == "false positive"].shape[0]
        count_false_negative = metaphor_df[metaphor_df.type == "false negative"].shape[0]
        count_true_positive = metaphor_df[metaphor_df.type == "true positive"].shape[0]
        count_true_negative = metaphor_df[metaphor_df.type == "true negative"].shape[0]
        count_unknown = metaphor_df[metaphor_df.type == "unknown"].shape[0]
        print("count_false_positive:", count_false_positive)
        print("count_false_negative:", count_false_negative)
        print("count_true_positive:", count_true_positive)
        print("count_true_negative:", count_true_negative)
        print("count_unknown", count_unknown)

    else:
        print("The candidate finder or the metaphor labeler is incorrect")

print("--- %s seconds ---" % (time.time() - start_time))

# #Hash table creation
# metaphorRegistry = Registry()
# metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
# metaphorRegistry.addMLabeler("cluster", clusteringFunction)
# metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
# metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
#
#
# # Parsing the command line
# args = parseCommandLine()
# cFinderFunction = metaphorRegistry.getCFinder(args.cfinder)
# mLabelerFunction = metaphorRegistry.getMLabeler(args.mlabeler)
#
#
# # Extracting data from text file
# textfile_path = "./data/TroFiDataset.txt"
# regex_dict = {
#     'verb': re.compile(r'\*{3}[a-z]+\*{3}'),
#     'cluster': re.compile(r'\*(non)?literal cluster\*'),
#     'id': re.compile(r'wsj[0-9]{2}:[0-9]+\s'),
#     'status': re.compile(r'[UNL]\W'),
#     'text': re.compile(r'[A-Z][^\s].+\.\/\.')
# }
#
# data = parse_file(textfile_path, regex_dict)
# data.to_csv("./data/TroFiDataset.csv", index=False)
#
# verbs = set(data["verb"].tolist())
#
# # Identifying the metaphors
# metaphor_list = []
# for i in range(data.shape[0]):
#     text = data.iloc[i]["text"]
#     # Object declaration
#     object = MetaphorIdentification(text)
#
#     # Annotating the text
#     object.annotateText()
#     object.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
#     object.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
#     if args.verbose:
#         print(object.getAnnotatedText())
#
#     # Finding candidates
#     object.findCandidates(cFinderFunction)
#     if args.verbose:
#         print(object.getCandidates())
#
#     # labeling Metaphors
#     object.labelMetaphors(mLabelerFunction, args.verbose)
#     if args.verbose:
#         print(object.getMetaphors())
#
#     # Find a type to each labeled metaphor (TP, TN, FP, FN)
#     for met in object.getMetaphors():
#         type = TroFiVerification(met, data.iloc[i])
#         metaphor = {'text': text, 'target': met.getTarget(), 'source': met.getSource(), 'result': met.getResult(), 'confidence': met.getConfidence(), 'type': type}
#         metaphor_list.append(metaphor)
#
#     # Progess of the algorithm
#     print(i, "out of", data.shape[0])
#
# metaphor_df = pd.DataFrame(metaphor_list)
# write_csv_path = "./data/TroFiMetaphors.csv"
# metaphor_df.to_csv(path_or_buf=write_csv_path, index=False)
#
# # Displaying the results
# count_false_positive = metaphor_df[metaphor_df.type == "false positive"].shape[0]
# count_false_negative = metaphor_df[metaphor_df.type == "false negative"].shape[0]
# count_true_positive = metaphor_df[metaphor_df.type == "true positive"].shape[0]
# count_true_negative = metaphor_df[metaphor_df.type == "true negative"].shape[0]
# count_unknown = metaphor_df[metaphor_df.type == "unknown"].shape[0]
# print("count_false_positive:", count_false_positive)
# print("count_false_negative:", count_false_negative)
# print("count_true_positive:", count_true_positive)
# print("count_true_negative:", count_true_negative)
# print("count_unknown", count_unknown)
