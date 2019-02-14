from .datastructs.annotated_text import AnnotatedText
from .sample_functions import posFunction, lemmatizingFunction

class MetaphorIdentification:
    def __init__(self, text=""):
        self.rawText = text
        self.annotatedText = AnnotatedText(self.rawText)
        self.candidates = None
        self.metaphors = []

    def getRawText(self):
        return self.rawText

    def getAnnotatedText(self):
        return self.annotatedText

    def getCandidates(self):
        return self.candidates

    def getMetaphors(self):
        return self.metaphors

    def annotTextAddColumn(self, name, annotationFunc):
        newCol = annotationFunc(self.annotatedText)
        self.annotatedText.addColumn(name, newCol)

    def findCandidates(self, identificationFunction):
        self.candidates = identificationFunction(self.annotatedText)

    def labelMetaphors(self, identificationFunction):
        self.metaphors = identificationFunction(self.candidates)

    def identification(self, cFinderFunction, mLabelerFunction, verbose):
        self.annotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
        self.annotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
        if verbose:
            print(self.getAnnotatedText())

        self.findCandidates(cFinderFunction)
        if verbose:
            print(self.getCandidates())

        self.labelMetaphors(mLabelerFunction)
        if verbose:
            print(self.getMetaphors())

        return self.getMetaphors()