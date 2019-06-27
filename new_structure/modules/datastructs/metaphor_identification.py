from new_structure.modules.datastructs.annotated_text import AnnotatedText
from new_structure.modules.datastructs.metaphor_group import MetaphorGroup
from new_structure.modules.datastructs.candidate_group import CandidateGroup
from new_structure.modules.datastructs.candidate import Candidate
import editdistance

class MetaphorIdentification:
    def __init__(self):
        self.mLabeler = {}
        self.cFinder = {}

        self.rawTexts = []
        self.annotatedTexts = {}
        self.candidates = {}
        self.metaphors = {}


        # rawTexts = list of strings (each string is a text in which we want to find metaphors)
        # annotatedTexts = dictionary: key = numbers (index of string in rawTexts), values = AnnotatedText objects
        # candidates = dictionary: key = cFinder IDs, values = dictionaries
        # -> dictionaries: key = numbers (index of string in rawTexts), values = CandidateGroup objects
        # metaphors = dictionary: key = mlabeler IDs, values = dictionaries
        # -> dictionaries: key = numbers (index of string in rawTexts), values = MetaphorGroup objects


    ### Procedures concerning the metaphor labelers and the candidate finders
    def addMLabeler(self, id, function):
        self.mLabeler[id] = function


    def addCFinder(self, id, function):
        self.cFinder[id] = function


    def getMLabeler(self, id):
        return self.mLabeler[id]


    def getCFinder(self, id):
        return self.cFinder[id]


    def isMLabeler(self, id):
        return id in self.mLabeler


    def isCFinder(self, id):
        return id in self.cFinder


    ### Procedures concerning one element
    def addText(self, text):
        if isinstance(text, list):
            for t in text:
                self.rawTexts.append(t)
        else:
            self.rawTexts.append(text)


    def getRawText(self, index=0):
        return self.rawTexts[index]


    def getAnnotatedText(self, index=0):
        return self.annotatedTexts.get(index, "")


    def getCandidates(self, index=0):
        return self.candidates[index] # What to return in case index is not in dictionary?


    def getMetaphors(self, index=0):
        return self.metaphors[index] # What to return in case index is not in dictionary?


    def annotateText(self, index=0):
        '''Annotate the raw text of index index'''
        self.annotatedTexts[index] = AnnotatedText(self.rawTexts[index])


    def annotTextAddColumn(self, name, annotationFunc, index=0):
        '''Add a column filled with the result of annotationFunc in the annotatedText of index index'''
        newCol = annotationFunc(self.annotatedTexts[index])
        self.annotatedTexts[index].addColumn(name, newCol)


    def findCandidates(self, cfinder_id, index=0):
        '''Apply the identificationFunction to find candidates in the annotated text of index index'''
        identificationFunction = self.getCFinder(cfinder_id)
        return identificationFunction(self.annotatedTexts[index])


    def labelMetaphors(self, mlabeler_id, cfinder_id, cand_type, index=0, verbose=False):
        '''Apply the labelingFunction to create metaphors from candidates in the candidateGroup of index index'''
        labelingFunction = self.getMLabeler(mlabeler_id)
        return labelingFunction(self.candidates[cfinder_id][index], cand_type, verbose)


    ### Procedures concerning dictionaries
    def getAllMetaphors(self, min_confidence = 0):
        '''Return an object MetaphorGroup containing the metaphors for all texts'''
        global_dic = dict()
        for k1, v1 in self.metaphors.items(): #for each mlabeler
            dic = dict()
            for k2, v2 in v1.items(): #v2 is a metaphorGroup
                v2 = v2.filterByConfidence(min_confidence)
                dic[k2] = str(v2)
            global_dic[k1] = dic
        return global_dic


    def annotateAllTexts(self):
        '''Annotate all the raw texts in the list rawText'''
        for index in range(len(self.rawTexts)):
            self.annotateText(index)


    def allAnnotTextAddColumn(self, name, annotationFunc):
        '''Add a column to all annotated texts in the dictionary annotatedTexts'''
        for index in range(len(self.annotatedTexts)):
            self.annotTextAddColumn(name, annotationFunc, index)


    def findAllCandidates(self, cfinder_id):
        '''Find the candidates in all annotated texts'''
        dic = self.candidates.get(cfinder_id, dict())

        for index in range(len(self.annotatedTexts)):
            dic[index] = self.findCandidates(cfinder_id, index)
            # self.findCandidates(identificationFunctionID, index)

        self.candidates[cfinder_id] = dic

    def labelAllMetaphors(self, mlabeler_id, cfinder_id, cand_type, verbose=False):
        '''Label all candidates in the dictionary Candidates'''
        dic = self.metaphors.get(mlabeler_id, dict())

        for index in range(len(self.candidates[cfinder_id])):
            print(index)
            dic[index] = self.labelMetaphors(mlabeler_id, cfinder_id, cand_type, index, verbose)

        self.metaphors[mlabeler_id] = dic


    def allCandidatesFromFile(self, source_list, target_list, label_list, indices_list):
        '''
            Create candidates from three lists
            Can be used to creates candidates from a CSV file which specify the candidates for a metaphor
        '''

        dic = self.candidates.get('fromFile', dict())

        # Test if the three lists have the same length
        # Modify for span > 1

        for i in range(len(indices_list)):
            text_index = indices_list[i]
            cand_gr = self.candidates.get(text_index, CandidateGroup())

            annotatedText = self.annotatedTexts[text_index]
            source = source_list[i].lower()
            target = target_list[i].lower()

            # Look for the source and the target in the annotated text
            sourceIndex, targetIndex = findPairInList(source, target, annotatedText.getColumn('word'))
            if sourceIndex == -1 or targetIndex == -1:
                sourceIndex, targetIndex = findPairInList(source, target, annotatedText.getColumn('lemma'))
            if sourceIndex == -1 or targetIndex == -1:
                sourceIndex, targetIndex = approximatePairInList(source, target, annotatedText.getColumn('lemma'))

            sourceSpan = (sourceIndex, sourceIndex)
            targetSpan = (targetIndex, targetIndex)

            new_candidate = Candidate(annotatedText, sourceIndex, sourceSpan, targetIndex, targetSpan)
            cand_gr.addCandidate(new_candidate)
            dic[text_index] = cand_gr
            # self.candidates[text_index] = cand_gr
        self.candidates['fromFile'] = dic


    ### Procedures concerning the comparison of different MLabelers

    def __getMetaphorsAsBooleans(self, ids, cfinder_id, cand_type, verbose, already_labeled):
        dic = dict()

        for mlabeler_id in ids:  # for each mlabeler
            if already_labeled == False:
                self.labelAllMetaphors(mlabeler_id, cfinder_id, cand_type, verbose)
            bool_dic = dict()
            for i, mg in self.metaphors[mlabeler_id].items():  # for each text
                bool_dic[i] = mg.getResults()  # get results of the metaphors in this text: list of boolean
            dic[mlabeler_id] = bool_dic

        return dic


    def agreeMLabelers(self, ids, cfinder_id, cand_type, verbose=False, already_labeled=False):
        '''
            Returns a dictionary indexed by texts.
            Each value is a list of booleans. The length of the list is equal to the number
            of possible metaphors (number of candidates) in the text.
            The booleans are True if all mlabeler agree on the result (metaphorical or literal),
            they are False if at least one of the mlabelers disagree with the others.
        '''
        dic = dict()
        agree = dict()

        if len(self.candidates[cfinder_id]) == 0:
            pass
            # Need to take care of this
        else:
            # for mlabeler_id in list(set(ids)): # for each mlabeler
            #     if already_labeled == False:
            #         self.labelAllMetaphors(mlabeler_id, cfinder_id, cand_type, verbose)
            #     bool_dic = dict()
            #     for i, mg in self.metaphors[mlabeler_id].items(): # for each text
            #         bool_dic[i] = mg.getResults()                 # get results of the metaphors in this text: list of boolean
            #     dic[mlabeler_id] = bool_dic
            dic = self.__getMetaphorsAsBooleans(ids, cfinder_id, cand_type, verbose, already_labeled)

            for i in range(len(self.candidates[cfinder_id])):  # for each text
                bool_list = list()
                first_mlabeler = dic[ids[0]]
                for j in range(len(first_mlabeler[i])):        # for each metaphor
                    tmp = list()
                    for mlabeler_id in ids:                    # for each mlabeler
                        tmp.append(dic[mlabeler_id][i][j]) # tmp contains the results for each method
                    n = len(set(tmp))                      # number of unique values in the list tmp
                    if n == 1:                             # If n=1 it means that all mlabelers agree
                        bool_list.append(True)
                    else:
                        bool_list.append(False)
                agree[i] = bool_list[:]

            return agree


    def percentageOfMetaphorical(self, ids, cfinder_id, cand_type, verbose=False, already_labeled=False):
        N_ids = len(ids)
        dic = dict()
        percent = dict()

        if len(self.candidates[cfinder_id]) == 0:
            pass
            # Need to take care of this
        else:
            # for mlabeler_id in ids:  # for each mlabeler
            #     if already_labeled == False:
            #         self.labelAllMetaphors(mlabeler_id, cfinder_id, cand_type, verbose)
            #     bool_dic = dict()
            #     for i, mg in self.metaphors[mlabeler_id].items():  # for each text
            #         bool_dic[i] = mg.getResults()  # get results of the metaphors in this text: list of boolean
            #     dic[mlabeler_id] = bool_dic
            dic = self.__getMetaphorsAsBooleans(ids, cfinder_id, cand_type, verbose, already_labeled)

            for i in range(len(self.candidates[cfinder_id])):  # for each text
                tmp_percent = list()
                first_mlabeler = dic[ids[0]]
                for j in range(len(first_mlabeler[i])):  # for each metaphor
                    # tmp = list()
                    count_True = 0
                    for mlabeler_id in ids:  # for each mlabeler
                        if dic[mlabeler_id][i][j] == True:
                            count_True += 1
                    tmp_percent.append(count_True / N_ids)
                percent[i] = tmp_percent[:]

            return percent


    def cohenKappa(self, mlabeler_id_1, mlabeler_id_2, cfinder_id, cand_type, verbose=False, already_labeled=False):
        # p0 = n_agreement / n_metaphors
        agree_dict = self.agreeMLabelers([mlabeler_id_1, mlabeler_id_2], cfinder_id, cand_type, verbose, already_labeled)
        count_agree, count_metaphors = 0,0
        for k, v in agree_dict.items(): # for each text
            count_metaphors += len(v)
            count_agree += v.count(True)
        p0 = count_agree / count_metaphors

        # pe = pTrue + pFalse
        # pTrue = (n_True_ml1 * n_True_ml2) / n_metaphors^2
        # pFalse = (n_False_ml1 * n_False_ml2) / n_metaphors^2
        n_True_ml1, n_True_ml2, n_False_ml1, n_False_ml2 = 0,0,0,0
        ml1, ml2 = self.metaphors[mlabeler_id_1], self.metaphors[mlabeler_id_2]

        for i, mg in ml1.items(): # for each text
            r = mg.getResults()
            n_True_ml1 += r.count(True)
            n_False_ml1 += r.count(False)

        for i, mg in ml2.items(): # for each text
            r = mg.getResults()
            n_True_ml2 += r.count(True)
            n_False_ml2 += r.count(False)

        pTrue = (n_True_ml1 * n_True_ml2) / count_metaphors**2
        pFalse = (n_False_ml1 * n_False_ml2) / count_metaphors**2
        pe = pTrue + pFalse

        k = (p0 - pe) / (1 - pe)
        return k




    ### Procedures concerning the evalutation of the Metaphor Labelers

    # 1: Set the candidates as the one in data set
    # 2: Label metaphors
    # 3: Compare to labels from data set
    def eval_mlabeler(self, ):
        pass


def findPairInList(w1, w2, word_list):
    w1_index = -1
    w2_index = -1

    w1.lower()
    w2.lower()
    for i in range(len(word_list)):
        word = word_list[i]
        word.lower()
        if word == w1:
            w1_index = i
        elif word == w2:
            w2_index = i

    return (w1_index, w2_index)


def approximatePairInList(w1, w2, word_list):
    w0 = word_list[0]
    w0.lower()

    w1.lower()
    w2.lower()
    w1_index = 0
    w2_index = 0
    w1_diff = editdistance.eval(w1, w0)
    w2_diff = editdistance.eval(w2, w0)

    for i in range(len(word_list)):
        word = word_list[i]
        word.lower()
        new_diff = editdistance.eval(w1, word)
        if new_diff < w1_diff:
            w1_diff = new_diff
            w1_index = i
        new_diff = editdistance.eval(w2, word)
        if new_diff < w2_diff:
            w2_diff = new_diff
            w2_index = i

    return (w1_index, w2_index)