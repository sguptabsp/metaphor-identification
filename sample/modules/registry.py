class Registry:

    def __init__(self):
        self.methods = {}
        self.candidates = {}

    def addMetLabeler(self, id, function):
        self.methods[id] = function

    def addCandFinder(self, id, function):
        self.candidates[id] = function

    def getMetLabeler(self, id):
        return self.methods[id]

    def getCandFinder(self, id):
        return self.candidates[id]

metaphorRegistry = Registry()