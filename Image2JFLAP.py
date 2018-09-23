import Image2V as I2V
import MyLibrary.JFLAPUtil as JUtil
def main():
    from sys import argv
    s, e = I2V.processImage(argv[1])

    xmlJff = JUtil.transitionToJFLAP(s, e)
    oFile = open("./Workspace/imgResult.jff", 'w')
    oFile.write(xmlJff)
    oFile.close

if __name__ == '__main__':
    main()