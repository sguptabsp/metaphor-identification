from tqdm import tqdm
import pandas as pd

ADJ_FILES = [ ("data/an2.txt", 2), ("data/an3.txt", 3), ("data/an4.txt", 4), ("data/an5.txt", 5)]
VERB_FILES = [("data/vn2.txt", 2)] # This file is a filtered version of vn2_v1: it only keep the lines with frequency > 1000
NGRAMS_FILES = {"adjNoun": ADJ_FILES, "verbNoun": VERB_FILES}



# an2 = pd.read_csv(ADJ_FILES[0][0], sep='\t', names=['Freq', 'Adj', 'Noun', 'Pos1', 'Pos2'])
#
# alphabet = dict()
#
# for i in tqdm(range(an2.shape[0])):
#     letter = str(an2['Adj'].iloc[i]).lower()[0]
#     index = alphabet.get(letter, [])
#     index.append(i)
#     alphabet[letter] = index
#
# for letter in alphabet:
#     index = alphabet[letter]
#     an2.iloc[index].to_csv('an2_' + letter + '.txt', sep='\t', header=None, index=False)

# ============================================================================================== #

# an3 = pd.read_csv(ADJ_FILES[1][0], sep='\t', names=['Freq', 'Word1', 'Word2', 'Word3', 'Pos1', 'Pos2', 'Pos3'])
#
# POS = ['Pos1', 'Pos2', 'Pos3']
#
# alphabet = dict()
#
# for i in range(an3.shape[0]):
#     l = list()
#     for p in POS:
#         if an3[p].iloc[i] == 'jj':
#             l.append((i, p))
#
#     for m in l:
#         ind = m[0]
#         pos = m[1]
#         word = 'Word' + pos[-1]
#
#         letter = str(an3[word].iloc[ind]).lower()[0]
#         index = alphabet.get(letter, [])
#         index.append(ind)
#         alphabet[letter] = index
#
# for letter in alphabet:
#     index = alphabet[letter]
#     an3.iloc[index].to_csv('an3_' + letter + '.txt', sep='\t', header=None, index=False)

# ============================================================================================== #

# an4 = pd.read_csv(ADJ_FILES[2][0], sep='\t', names=['Freq', 'Word1', 'Word2', 'Word3', 'Word4', 'Pos1', 'Pos2', 'Pos3', 'Pos4'])
#
# POS = ['Pos1', 'Pos2', 'Pos3', 'Pos4']
#
# alphabet = dict()
#
# for i in range(an4.shape[0]):
#     l = list()
#     for p in POS:
#         if an4[p].iloc[i] == 'jj':
#             l.append((i, p))
#
#     for m in l:
#         ind = m[0]
#         pos = m[1]
#         word = 'Word' + pos[-1]
#
#         letter = str(an4[word].iloc[ind]).lower()[0]
#         index = alphabet.get(letter, [])
#         index.append(ind)
#         alphabet[letter] = index
#
# for letter in alphabet:
#     index = alphabet[letter]
#     an4.iloc[index].to_csv('an4_' + letter + '.txt', sep='\t', header=None, index=False)

# ============================================================================================== #

# an5 = pd.read_csv(ADJ_FILES[3][0], sep='\t', names=['Freq', 'Word1', 'Word2', 'Word3', 'Word4', 'Word5', 'Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5'])
#
# POS = ['Pos1', 'Pos2', 'Pos3', 'Pos4', 'Pos5']
#
# alphabet = dict()
#
# for i in range(an5.shape[0]):
#     l = list()
#     for p in POS:
#         if an5[p].iloc[i] == 'jj':
#             l.append((i, p))
#
#     for m in l:
#         ind = m[0]
#         pos = m[1]
#         word = 'Word' + pos[-1]
#
#         letter = str(an5[word].iloc[ind]).lower()[0]
#         index = alphabet.get(letter, [])
#         index.append(ind)
#         alphabet[letter] = index
#
# for letter in alphabet:
#     index = alphabet[letter]
#     an5.iloc[index].to_csv('an5_' + letter + '.txt', sep='\t', header=None, index=False)

# ============================================================================================== #

vn2 = pd.read_csv(VERB_FILES[0][0], sep='\t', names=['Freq', 'Verb', 'Noun', 'Pos1', 'Pos2'])

alphabet = dict()

for i in tqdm(range(vn2.shape[0])):
    letter = str(vn2['Verb'].iloc[i]).lower()[0]
    index = alphabet.get(letter, [])
    index.append(i)
    alphabet[letter] = index

for letter in alphabet:
    index = alphabet[letter]
    vn2.iloc[index].to_csv('vn2_' + letter + '.txt', sep='\t', header=None, index=False)