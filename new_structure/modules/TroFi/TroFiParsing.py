import re
import pandas as pd


def parse_line(line, regex_dict):
    """
    Do a regex search against all defined regexes and
    return a dictionnary: key -> matched text

    """
    matches = {}

    for key, rx in regex_dict.items():
        match = rx.search(line)
        if match:
            matches[key] = match.group()

    return matches

def parse_file(filepath, regex_dict):
    data = []

    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for matches with a regex dictionnary
            matches = parse_line(line, regex_dict)
            text = ""

            for key, value in matches.items():
                # extract verb
                if key == 'verb':
                    verb = value[3:-3] #removing the ***

                # extract cluster
                elif key == 'cluster':
                    cluster = value[1:-9] #removing the * and the word cluster

                # extract id
                elif key == 'id':
                    id = value[0:-1] #removing the \t at the end

                # extract status
                elif key == 'status':
                    status = value[0:-1] #removing the \t at the end

                #extract text
                elif key == 'text':
                    text = value[0:-3] #removing the "./." at the end

            if text != "":
                # create a dictionary containing this row of data
                row = {
                    'verb': verb,
                    'cluster': cluster,
                    'id': id,
                    'status': status,
                    'text': text,
                }
                # append the dictionary to the data list
                data.append(row)

            line = file_object.readline()

        # create a pandas DataFrame from the list of dicts
        data = pd.DataFrame(data)
        # reordering the columns
        data = data[['verb', 'cluster', 'id', 'status', 'text']]

    return data



# textfile_path = "../data/TroFiMetaphorDataset.txt"
# csvfile_path = "../data/TroFiMetaphorDataset.csv"
#
# regex_dict = {
#     'verb': re.compile(r'\*{3}[a-z]+\*{3}'),
#     'cluster': re.compile(r'\*(non)?literal cluster\*'),
#     'id': re.compile(r'wsj[0-9]{2}:[0-9]+\s'),
#     'status': re.compile(r'[UNL]\W'),
#     'text': re.compile(r'[A-Z][^\s].+\.\/\.')
# }
#
# data = parse_file(textfile_path, regex_dict)
# data.to_csv(path_or_buf=csvfile_path, index=False)