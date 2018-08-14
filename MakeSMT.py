class State:
    def __init__(self):
        self.zero = None
        self.one = None
    
    def isComplete(self):
        return (self.zero != None) and (self.one != None)

    def setZero(self, zeroState):
        self.zero = zeroState

    def setOne(self, oneState):
        self.one = oneState