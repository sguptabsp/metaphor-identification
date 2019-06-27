import csv
import argparse as ap
import pandas as pd

AT_PATH = "data/annotated_corpus.csv"
MET_PATH = "data/results.csv"
SAMPLE_PATH = "data/sample.txt"
DEFAULT_TEXT = "The original text is divided into three or more parts - and some new punctuation of course. We can talk about a sweet child and a tall child. A golden boy and a young man. And also red green brain ideas and intelligent boats. The mouse eats a cat and the man attacked a castle. "

# ALL = False

def writeToCSV(dicList, path, columns):
	with open(path, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, columns)
		writer.writeheader()
		writer.writerows(dicList)

def readFromTextFile(path):
	data = ""
	with open(path, 'r', encoding='utf8') as textFile:
		data = textFile.read()
	return data

def readDataFrame(DF, cgenerator):
	col = DF.columns.tolist()

	texts = list()
	for c in ['text', 'Text', 'sentence', 'Sentence']:
		if c in col:
			texts = DF[c].values.tolist()
			break

	sources = list()
	targets = list()
	if cgenerator:
		for c in ['source', 'sources', 'Source', 'Sources']:
			if c in col:
				sources = DF[c].values.tolist()
				break
		for c in ['target', 'targets', 'Target', 'Targets']:
			if c in col:
				targets = DF[c].values.tolist()
				break
	return (texts, sources, targets)


def readFromCsvFile(path, cgenerator):
	data = pd.read_csv(path)
	return readDataFrame(data, cgenerator)


def readFromExcelFile(path, cgenerator):
	data = pd.read_excel(path)
	return readDataFrame(data, cgenerator)


def extractText(path, cgenerator):
	path.lower()
	if path.endswith('.txt'):
		return readFromTextFile(path)
	elif path.endswith('.csv'):
		return readFromCsvFile(path, cgenerator)
	elif path.endswith('.xlsx'):
		return readFromExcelFile(path, cgenerator)
	else:
		print("Does not handle this file format")


def getText(args):
	if args.file:
		return extractText(args.file, args.cgenerator)
	elif args.string:
		return (args.string, [], [])
	else:
		return (DEFAULT_TEXT, [], [])

def parseCommandLine():
	parser = ap.ArgumentParser()
	parser.add_argument("-v", "--verbose", default=False,
						help="print details", action="store_true")
	# parser.add_argument("-a", "--all", help="use every available methods of identification and labeling",
	# 					action="store_true")
	parser.add_argument("-ml", "--mlabeler", type=str, default="darkthoughts",
						help="choose the metaphor labeling method: darkthoughts, cluster")
	parser.add_argument("-cf", "--cfinder", type=str, default="adjNoun",
						help="choose the candidate finding method: adjNoun, verbNoun")
	parser.add_argument("-cg", "--cgenerator", default=False, action="store_true", help="Generate candidates from an excel files")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("-f", "--file", type=str, help="look for metaphors in a text file")
	group.add_argument("-s", "--string", type=str, help="look for metaphors in a specified string")
	args = parser.parse_args()

	return args



### Old functions

# def getText(path):
# 	data = ""
# 	with open(path, 'r') as textFile:
# 		data = textFile.read()
# 	return data

# def parseCommandLine():
# 	global VERBOSE
# 	global ALL
# 	global M_DARKTHOUGHT
# 	global M_CLUSTERING
# 	global I_ADJNOUN
# 	global I_VERBNOUN
# 	parser = ap.ArgumentParser()
# 	parser.add_argument("-v", "--verbose", help="print details", action="store_true")
# 	parser.add_argument("-a", "--all", help="use every available methods of identification and labeling", action="store_true")
# 	parser.add_argument("-m", "--method", type=int, help="choose the metaphor identifying method: 1-Darkthought, 2-Clustering")
# 	parser.add_argument("-i", "--id", type=int, help="choose the candidate identifying method: 1-AdjNoun, 2-VerbNoun")
# 	group = parser.add_mutually_exclusive_group()
# 	group.add_argument("-f", "--file", type=str, help="look for metaphors in a text file")
# 	group.add_argument("-s", "--string", type=str, help="look for metaphors in a specified string")
# 	args = parser.parse_args()
#
# 	if args.verbose:
# 		VERBOSE = True
#
# 	if args.all:
# 		ALL = True
# 	else:
# 		ALL = False
# 		if args.method == 1:
# 			M_DARKTHOUGHT = True
# 			M_CLUSTERING = False
# 		elif args.method == 2:
# 			M_DARKTHOUGHT = False
# 			M_CLUSTERING = True
#
# 		if args.id == 1:
# 			I_ADJNOUN = True
# 			I_VERBNOUN = False
# 		elif args.id == 2:
# 			I_ADJNOUN = False
# 			I_VERBNOUN = True
#
# 	if args.file:
# 		return getText(args.file)
# 	elif args.string:
# 		return args.string
# 	else:
# 		return DEFAULT_TEXT
