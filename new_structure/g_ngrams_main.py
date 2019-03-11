import re
import pandas as pd
from new_structure.modules.g_ngrams import parse_file

folder_path = "../../Google_ngrams"
# file_name = "test"
file_names = ["googlebooks-eng-all-2gram-20120701-aa",
              "googlebooks-eng-all-2gram-20120701-ab",
              "googlebooks-eng-all-2gram-20120701-ac",
              "googlebooks-eng-all-2gram-20120701-ad",
              "googlebooks-eng-all-2gram-20120701-ae",
              "googlebooks-eng-all-2gram-20120701-af",
              "googlebooks-eng-all-2gram-20120701-ag",
              "googlebooks-eng-all-2gram-20120701-ah",
              "googlebooks-eng-all-2gram-20120701-ai",
              "googlebooks-eng-all-2gram-20120701-aj",
              "googlebooks-eng-all-2gram-20120701-ak",
              "googlebooks-eng-all-2gram-20120701-al",
              "googlebooks-eng-all-2gram-20120701-am",
              "googlebooks-eng-all-2gram-20120701-an",
              "googlebooks-eng-all-2gram-20120701-ao",
              "googlebooks-eng-all-2gram-20120701-ap",
              "googlebooks-eng-all-2gram-20120701-aq",
              "googlebooks-eng-all-2gram-20120701-ar",
              "googlebooks-eng-all-2gram-20120701-as",
              "googlebooks-eng-all-2gram-20120701-at",
              "googlebooks-eng-all-2gram-20120701-au",
              "googlebooks-eng-all-2gram-20120701-av",
              "googlebooks-eng-all-2gram-20120701-aw",
              "googlebooks-eng-all-2gram-20120701-ax",
              "googlebooks-eng-all-2gram-20120701-ay",
              "googlebooks-eng-all-2gram-20120701-az",
              ]

regex_dict = {
    'verb': re.compile(r'[a-zA-Z]+_VERB'),
    'noun': re.compile(r'[a-zA-Z]+_NOUN'),
}


## Section 1: from google txt file to csv ----------------------------------------------------------------------------------

# for i in range(0,len(file_names)):
#     data = parse_file(folder_path + "/" + file_names[i], regex_dict)
#     print(data)
#     df = pd.DataFrame(data)
#     df.to_csv("./data/darkthoughts/" + file_names[i] + ".csv", index=False)



## Section 2: from csv to text ---------------------------------------------------------------------------------------------

df = pd.read_csv("./data/darkthoughts/" + "googlebooks-eng-all-2gram-20120701-aa" + ".csv")

df['pos_v'] = "v"
df['pos_n'] = "n"
df = df[["count", "verb", "noun", "pos_v", "pos_n"]]

df.to_csv("./data/darkthoughts/" + "googlebooks-eng-all-2gram-20120701-aa" + ".txt", header=None, index=None, sep="\t")