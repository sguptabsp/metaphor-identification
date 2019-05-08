def parse_line(line, regex_dict):
    s = line.split()
    w1 = s[0]
    w2 = s[1]
    y = s[2]
    mc = s[3]

    if regex_dict['verb'].fullmatch(w1) and regex_dict['noun'].fullmatch(w2):
        matches = {'verb': w1[:-5], 'noun': w2[:-5], 'year': int(y), 'match_count': int(mc)}
    if regex_dict['noun'].fullmatch(w1) and regex_dict['verb'].fullmatch(w2):
        matches = {'noun': w1[:-5], 'verb': w2[:-5], 'year': int(y), 'count': int(mc)}
    else:
        matches = None

    return matches

def parse_file(filepath, regex_dict):
    previous_noun = ""
    previous_verb = ""
    total_count = 0
    data = []

    # open the file and read through it line by line
    with open(filepath, 'r', encoding="utf8") as file_object:
        line = file_object.readline()
        while line:
            # at each line check for matches with a regex dictionnary
            matches = parse_line(line, regex_dict)
            if matches != None:
                if matches['noun'] != previous_noun and matches['verb'] != previous_verb:
                    row = {
                        'verb': previous_verb,
                        'noun': previous_noun,
                        'count': total_count,
                    }
                    data.append(row)

                    previous_verb = matches['verb']
                    previous_noun = matches['noun']
                    total_count = matches['count']
                else:
                    total_count += matches['count']

            line = file_object.readline()

        row = {
            'verb': previous_verb,
            'noun': previous_noun,
            'count': total_count,
        }
        data.append(row)

    return data[1:]

