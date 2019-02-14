# Author : Thomas Buffagni
# Latest revision : 02/14/2019

import modules.utils as utils
from modules.sample_functions import verbNounFinder, adjNounFinder
from modules.darkthoughts import darkthoughtsFunction
from modules.cluster_module import clusteringFunction
from modules.registry import Registry
from modules.MetaphorIdentification import MetaphorIdentification

if __name__ == "__main__":

	metaphorRegistry = Registry()
	metaphorRegistry.addMLabeler("darkthoughts", darkthoughtsFunction)
	metaphorRegistry.addMLabeler("cluster", clusteringFunction)
	metaphorRegistry.addCFinder("verbNoun", verbNounFinder)
	metaphorRegistry.addCFinder("adjNoun", adjNounFinder)

	text = utils.getText(utils.args)
	cFinderFunction = metaphorRegistry.getCFinder(utils.args.cfinder)
	mLabelerFunction = metaphorRegistry.getMLabeler(utils.args.mlabeler)

	object = MetaphorIdentification(text)
	met = object.identification(cFinderFunction, mLabelerFunction, utils.args.verbose )
