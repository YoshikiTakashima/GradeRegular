import Image2V as I2V
import JFLAP2V as J2V
import MyLibrary.VerilogUtil as VUtil

def main():
    from sys import argv
    file1 = argv[1]
    v1 = ""
    file2 = argv[2]
    v2 = ""

    if ".jpg" in file1:
        v1s, v1e = I2V.processImage(file1)
        v1 = VUtil.transitionToVerilog(v1s, v1e)
    elif ".jff" in file1:
        v1 = J2V.processJFLAP(file1)
    
    if ".jpg" in file2:
        v2s, v2e = I2V.processImage(file2)
        v2 = VUtil.transitionToVerilog(v2s, v2e)
    elif ".jff" in file2:
        v2 = J2V.processJFLAP(file2)

    i = v1.find("Automaton")
    v1 = v1[:i] + "Automaton1" + v1[i + len("Automaton"):]

    i = v2.find("Automaton")
    v2 = v2[:i] + "Automaton2" + v2[i + len("Automaton"):]

    verilog = v1 + v2 + VUtil.EQUALTEMPLATE

    fOut = open("./Workspace/toFormal.v", "w")
    fOut.write(verilog)
    fOut.close()

if __name__ == '__main__':
    main()