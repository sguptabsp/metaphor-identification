from new_structure.modules.datastructs.annotated_text import AnnotatedText

class MetaphorIdentification:
    def __init__(self, text=""):
        self.rawText = text
        self.annotatedText = None
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

    def annotateText(self):
        self.annotatedText = AnnotatedText(self.rawText)

    def annotTextAddColumn(self, name, annotationFunc):
        newCol = annotationFunc(self.annotatedText)
        self.annotatedText.addColumn(name, newCol)

    def findCandidates(self, identificationFunction):
        self.candidates = identificationFunction(self.annotatedText)
        print(self.candidates)

    def labelMetaphors(self, identificationFunction, cand_type, verbose=False):
        self.metaphors = identificationFunction(self.candidates, cand_type, verbose)