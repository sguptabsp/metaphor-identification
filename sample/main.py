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

idFunctions = [sf.adjNounFinder, sf.verbNounFinder]

if __name__ == "__main__":

	data = utils.getText(utils.args)
	annotator = an.Annotator(data)
	annotator.addColumn("POS", sf.posFunction)           #Set a part-of-speech to each word of the string
	annotator.addColumn("lemma", sf.lemmatizingFunction) #Set a lemma to each word of the string
	annotatedText = annotator.getAnnotatedText()
	annotatedText.writeToCSV(utils.AT_PATH)
	if utils.args.verbose:
		print(annotatedText)

	identifier = ci.CandidateIdentifier(annotatedText)
	identifier.IDCandidates(metaphorRegistry.getCandidate(utils.args.id))
	candidates = identifier.getCandidates()
	if utils.args.verbose:
		print(candidates)

	labeler = mi.MetaphorIdentifier(candidates)
	# labeler.IDMetaphors(sf.testLabelFunction)
	labeler.IDMetaphors(metaphorRegistry.getMethod(utils.args.method))
	results = labeler.getMetaphors()
	results.writeToCSV(utils.MET_PATH)
	print(results)

	
# read from csv file, take source/target and keep the rest