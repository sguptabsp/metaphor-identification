import re
import os
import pandas as pd
from new_structure.modules.g_ngrams import parse_file

folder_path = "../../Google_ngrams"


regex_dict = {
    'verb': re.compile(r'[a-zA-Z]+_VERB'),
    'noun': re.compile(r'[a-zA-Z]+_NOUN'),
}


file_names = os.listdir(folder_path)
print(file_names)

## Section 1: from google txt file to csv ----------------------------------------------------------------------------------

for i in range(0,len(file_names)):
    data = parse_file(folder_path + "/" + file_names[i], regex_dict)
    df = pd.DataFrame(data)
    df.to_csv("./data/darkthoughts/" + file_names[i] + ".csv", index=False)
    print(i, "out of", len(file_names))


## Section 2: from csv to text ---------------------------------------------------------------------------------------------

for i in range(0,len(file_names)):
    df = pd.read_csv("./data/darkthoughts/" + file_names[i] + ".csv")

    df['pos_v'] = "v"
    df['pos_n'] = "n"
    df = df[["count", "verb", "noun", "pos_v", "pos_n"]]

    df.to_csv("./data/darkthoughts/" + file_names[i] + ".txt", header=None, index=None, sep="\t")
    print(i, "out of", len(file_names))


## Section 3: tests ---------------------------------------------------------------------------------------------


from new_structure.modules.datastructs.ngrams import CollocationList, parseNgrams

for i in range(len(file_names)):
    path_vn = "./data/darkthoughts/" + file_names[i] + ".txt"
    collocations_vn = CollocationList()
    parseNgrams(collocations_vn, path_vn, 2, "v", "n")
    print(collocations_vn.collocations)
    print(i, "out of", len(file_names))

