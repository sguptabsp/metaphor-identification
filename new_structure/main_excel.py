import pandas as pd

from new_structure.modules.utils import parseCommandLine
from new_structure.modules.sample_functions import posFunction, lemmatizingFunction
#Candidate finding functions
from new_structure.modules.sample_functions import verbNounFinder, adjNounFinder
#Metaphor labeling functions
from new_structure.modules.darkthoughts import darkthoughtsFunction
from new_structure.modules.cluster_module import clusteringFunction
#Data structures
from new_structure.modules.datastructs.registry import Registry
from new_structure.modules.datastructs.metaphor_identification import MetaphorIdentification

# ----------------------------------------------------------------------------------------------------------------

from new_structure.modules.datastructs.candidate import Candidate
from new_structure.modules.datastructs.candidate_group import CandidateGroup

#Only works for source and target of length 1 !!!!!!!!!!!!!!!!!!!
def candidate_from_pair(annotatedText, source, target):
    source = source.lower()
    target = target.lower()
    sourceIndex = -1
    targetIndex = -1

    for cn in ['word','lemma']:
        column = annotatedText.getColumn(cn)

        for i in range(len(column)):
            e = column[i].lower()
            if e == source:
                sourceIndex = i
            elif e == target:
                targetIndex = i
        if sourceIndex != -1 and targetIndex != -1:
            break
        sourceIndex = -1
        targetIndex = -1

    # errors = []
    if sourceIndex == -1 and targetIndex == -1:
        raise ValueError("Source or Target is not in the annotated text")
        # errors.append((source, target))
    # print(errors)

    sourceSpan = (sourceIndex, sourceIndex)
    targetSpan = (targetIndex, targetIndex)

    return Candidate(annotatedText, sourceIndex, sourceSpan, targetIndex, targetSpan)

# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    args = parseCommandLine()

    # Hash table creation
    metaphorRegistry = Registry()
    metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
    metaphorRegistry.addCFinder("adjNoun", adjNounFinder)
    metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
    metaphorRegistry.addMLabeler("cluster", clusteringFunction)

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
        # print(data.columns)

        # Definition of names
        text_column = 'sentence'
        source_column = 'adj'
        target_column = 'noun'

        N = 5
        text_list = data[text_column].values.tolist()
        list = MetaphorIdentification(text_list[0:N])

        list.annotateAllTexts()
        list.allAnnotTextAddColumn("POS", posFunction)
        list.allAnnotTextAddColumn("lemma", lemmatizingFunction)

        # list.findAllCandidates(cFinderFunction)
        source_list = data[source_column].values.tolist()[0:N]
        target_list = data[target_column].values.tolist()[0:N]
        indices_list = [i for i in range(N)]
        list.allCandidatesFromColumns(source_list, target_list, indices_list)

        list.labelAllMetaphors(mLabelerFunction, args.cfinder)
        print(list.getAllMetaphors())
