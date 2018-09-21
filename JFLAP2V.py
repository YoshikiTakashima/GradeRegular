import xml.etree.ElementTree
from Image2V import transitionSetToStateValueMap
import MyLibrary.VerilogUtil as VUtil

def xmlRoot2Verilog(root):
    auto = root.findall("automaton")[0]
    # print(auto)

    states = []
    for s in auto.findall("state"):
        lbl = 'NA'
        x = int(round(float(s[0].text)))
        y = int(round(float(s[1].text)))
        init = s.findall("initial")
        final = s.findall("final")
        if len(final) > 0:
            lbl = 'A'
        
        if len(init) > 0:
            states = [(x, y, 1, lbl)] + states
        else:
            states.append((x, y, 1, lbl))
    
    transitions = []
    for t in auto.findall("transition"):
        frm = int(t[0].text)
        to = int(t[1].text)
        val = t[2].text
        # print("frm: {}, to: {}, val: {}".format(frm, to, val))

        val = t[2].text
        if val == '0':
            val = 0
        elif val == '1':
            val = 1
        else:
            val = 2
        transitions.append(((frm, -1), (to, val)))

    encoded = transitionSetToStateValueMap(states, transitions)
    return VUtil.transitionToVerilog(states, encoded)

def processJFLAP(path):
    e = xml.etree.ElementTree.parse(path).getroot()
    return(xmlRoot2Verilog(e))

def main():
    from sys import argv
    print(processJFLAP(argv[1]))


if __name__ == '__main__':
    main()