class Registry:

    def __init__(self):
        self.methods = {}
        self.candidates = {}

    def addMLabeler(self, id, function):
        self.methods[id] = function

    def addCFinder(self, id, function):
        self.candidates[id] = function

    def getMLabeler(self, id):
        return self.methods[id]

    def getCFinder(self, id):
        return self.candidates[id]