# Author : Henri Toussaint
# Latest revision : 01/25/2017

# 23/02 12pm


import modules.annotator as an
import modules.cand_id as ci
import modules.met_id as mi
import modules.sample_functions as sf
import modules.utils as utils
from modules.darkthoughts import darkthoughtsFunction
from modules.cluster_module import clusteringFunction
from modules.registry import metaphorRegistry

import pandas as pd

idFunctions = [sf.adjNounFinder, sf.verbNounFinder]

read_csv_path = "./data/TroFiMetaphorDataset.csv"
write_csv_path = "./data/TroFiMetaphorFullResults.csv"

if __name__ == "__main__":

    data = pd.read_csv(read_csv_path)
    result_list = []

    i=0
    for text in data["text"]:
        # print(text)
        data = text
        annotator = an.Annotator(data)
        annotator.addColumn("POS", sf.posFunction)  # Set a part-of-speech to each word of the string
        annotator.addColumn("lemma", sf.lemmatizingFunction)  # Set a lemma to each word of the string
        annotatedText = annotator.getAnnotatedText()
        annotatedText.writeToCSV(utils.AT_PATH)
        if utils.args.verbose:
            print(annotatedText)

        identifier = ci.CandidateFinder(annotatedText)
        identifier.FindCandidates(metaphorRegistry.getCFinder(utils.args.cfinder))
        candidates = identifier.getCandidates()
        if utils.args.verbose:
            print(candidates)

        labeler = mi.MetaphorLabeler(candidates)
        # labeler.IDMetaphors(sf.testLabelFunction)
        labeler.LabelMetaphors(metaphorRegistry.getMLabeler(utils.args.mlabeler))
        results = labeler.getMetaphors()

        for r in results:
            metaphor = {'text': text, 'target': r.getTarget(), 'source': r.getSource(), 'result': r.getResult(), 'confidence': r.getConfidence()}
            result_list.append(metaphor)
            print(i)
            i+=1
        # results.writeToCSV(utils.MET_PATH)
        # print(results)

    results = pd.DataFrame(result_list)
    print(results)
    results.to_csv(path_or_buf=write_csv_path, index=False)