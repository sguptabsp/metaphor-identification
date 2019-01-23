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

idFunctions = [sf.adjNounFinder, sf.verbNounFinder]

if __name__ == "__main__":

	data = utils.parseCommandLine()
	annotator = an.Annotator(data)
	annotator.addColumn("POS", sf.posFunction)
	annotator.addColumn("lemma", sf.lemmatizingFunction)
	annotatedText = annotator.getAnnotatedText()
	annotatedText.writeToCSV(utils.AT_PATH)
	if utils.VERBOSE:
		print(annotatedText)

	identifier = ci.CandidateIdentifier(annotatedText)
	if utils.I_ADJNOUN:
		identifier.IDCandidates(sf.adjNounFinder)
	elif utils.I_VERBNOUN:
		identifier.IDCandidates(sf.verbNounFinder)
	candidates = identifier.getCandidates()
	if utils.VERBOSE:
		print(candidates)

	labeler = mi.MetaphorIdentifier(candidates)
	# labeler.IDMetaphors(sf.testLabelFunction)
	if utils.M_DARKTHOUGHT:
		labeler.IDMetaphors(darkthoughtsFunction)
	elif utils.M_CLUSTERING:
		labeler.IDMetaphors(clusteringFunction)
	results = labeler.getMetaphors()
	results.writeToCSV(utils.MET_PATH)
	print(results)

	
# read from csv file, take source/targett and keep the rest