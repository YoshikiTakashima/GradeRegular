class Line:
    def __init__(self, cvLine):
        self.cvLine = cvLine
        self.value = -1
    
    def isReady(self):
        return (self.value >= 0) and (self.value < 2)

def recognize(image):
    pass

def main():
    pass

if __name__ == '__main__':
    main()