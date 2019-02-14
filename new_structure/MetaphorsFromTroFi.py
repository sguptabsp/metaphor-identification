# Author : Henri Toussaint
# Latest revision : 01/25/2017

# 23/02 12pm

import modules.utils as utils
from modules.darkthoughts import darkthoughtsFunction
from modules.cluster_module import clusteringFunction
from modules.registry import metaphorRegistry
from MIImplementation import mIImplementation

import pandas as pd


read_csv_path = "./data/TroFiMetaphorDataset.csv"
write_csv_path = "./data/TroFiMetaphorFullResults.csv"

if __name__ == "__main__":

    data = pd.read_csv(read_csv_path)
    result_list = []

    i=0
    for text in data["text"]:
        metaphors = mIImplementation(text,                                              # text we cant to analyse
                                     metaphorRegistry.getCFinder(utils.args.cfinder),   # candidate finder function (from command line)
                                     metaphorRegistry.getMLabeler(utils.args.mlabeler)) # metaphor labeling function (from command line)
        for met in metaphors:
            result_list.append(met)
            i += 1
        print(i)

