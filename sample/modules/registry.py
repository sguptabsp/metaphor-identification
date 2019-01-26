class Registry:

    def __init__(self):
        self.methods = {}
        self.candidates = {}

    def addMethod(self,id, function):
        self.methods[id] = function

    def addCandidate(self, id, function):
        self.candidates[id] = function

    def getMethod(self, id):
        return self.methods[id]

    def getCandidate(self, id):
        return self.candidates[id]

metaphorRegistry = Registry()