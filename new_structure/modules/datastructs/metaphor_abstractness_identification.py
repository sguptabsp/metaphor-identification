from new_structure.modules.datastructs.MetaphorIdentification import MetaphorIdentification
from new_structure.modules.datastructs.annotated_text import AnnotatedText
import pandas as pd

class MetaphorAbstractnessIdentification(MetaphorIdentification):
    def __init__(self, text=""):
        self.rawText = text
        super().__init__()

    def annotateText(self):
        self.annotatedText = None

    def annotTextAddColumn(self, name, annotationFunc):
        pass

    def findCandidates(self, identificationFunction):
        fields = ['adj', 'noun']
        MET_AN_EN_TEST = pd.read_excel('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx', sheetname='MET_AN_EN', usecols=fields)
        MET_AN_EN_TEST['class'] = 1
        LIT_AN_EN_TEST = pd.read_excel('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx', sheetname='LIT_AN_EN', usecols=fields)
        LIT_AN_EN_TEST['class'] = 0
        MET_AN_EN = pd.read_table('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
        MET_AN_EN['class'] = 1
        LIT_AN_EN = pd.read_table('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
        LIT_AN_EN['class'] = 0

        df = pd.concat([LIT_AN_EN_TEST, MET_AN_EN_TEST, LIT_AN_EN, MET_AN_EN])
        df = pd.DataFrame(df)

        self.candidates = df

    def labelMetaphors(self, identificationFunction, cand_type, verbose=False):
        self.metaphors = identificationFunction(self.candidates, cand_type, verbose)
