from new_structure.modules.datastructs.annotated_text import AnnotatedText
from new_structure.modules.datastructs.metaphor_group import MetaphorGroup
from new_structure.modules.datastructs.candidate_group import CandidateGroup
from new_structure.modules.datastructs.candidate import Candidate
import editdistance

class MetaphorIdentificationList:
    def __init__(self, texts):
        self.annotatedTexts = {}
        self.candidates = {}
        self.metaphors = {}
        if isinstance(texts, list):
            self.rawTexts = texts
        else:
            self.rawTexts = []
            self.rawTexts.append(texts)

    ### Procedures concerning one element
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

    def findCandidates(self, identificationFunction, index=0):
        '''Apply the identificationFunction to find candidates in the annotated text of index index'''
        self.candidates[index] = identificationFunction(self.annotatedTexts[index])

    def labelMetaphors(self, labelingFunction, cand_type, index=0, verbose=False):
        '''Apply the labelingFunction to create metaphors from candidates in the candidateGroup of index index'''
        self.metaphors[index] = labelingFunction(self.candidates[index], cand_type, verbose)

    ### Procedures concerning dictionaries
    def getAllMetaphors(self):
        '''Return an object MetaphorGroup containing the metaphors for all texts'''
        if len(self.metaphors) == 1:
            return self.metaphors[0]
        allMetaphors = MetaphorGroup()
        for metgr in self.metaphors.values():
            for i in range(metgr.getSize()):
                allMetaphors.addMetaphor(metgr.metaphors[i])
        return allMetaphors

    def annotateAllTexts(self):
        '''Annotate all the raw texts in the list rawText'''
        for index in range(len(self.rawTexts)):
            self.annotateText(index)

    def allAnnotTextAddColumn(self, name, annotationFunc):
        '''Add a column to all annotated texts in the dictionary annotatedTexts'''
        for index in range(len(self.annotatedTexts)):
            self.annotTextAddColumn(name, annotationFunc, index)

    def findAllCandidates(self, identificationFunction):
        '''Find the candidates in all annotated texts'''
        for index in range(len(self.annotatedTexts)):
            self.findCandidates(identificationFunction, index)

    def labelAllMetaphors(self, labelingFunction, cand_type, verbose=False):
        '''Label all candidates in the dictionary Candidates'''
        for index in range(len(self.candidates)):
            print(index)
            self.labelMetaphors(labelingFunction, cand_type, index, verbose)

    def allCandidatesFromColumns(self, source_list, target_list, indices_list):
        '''Create candidates from three lists
            Can be used to creates candidates from a CSV file which specify the candidates for a metaphor
        '''

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
            self.candidates[text_index] = cand_gr


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