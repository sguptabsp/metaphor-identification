class Registry:

    def __init__(self):
        self.mLabeler = {}
        self.cFinder = {}

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
