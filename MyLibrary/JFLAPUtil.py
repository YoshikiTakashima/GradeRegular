JFLAPTEMPLATE = \
"""<?xml version="1.0" encoding="UTF-8" standalone="no"?><!--Created with JFLAP 7.1.--><structure>&#13;
	<type>fa</type>&#13;
	<automaton>&#13;
		<!--The list of states.-->&#13;
		{}
		<!--The list of transitions.-->&#13;
		{}
	</automaton>&#13;
</structure>
"""

INITIAL = "<initial/>&#13;"
FINAL = "<final/>&#13;"
STATETEMPLATE = """
		<state id="{0}" name="q{0}">&#13;
			<x>{1}</x>&#13;
			<y>{2}</y>&#13;
			{3}
		</state>&#13;
"""
READTEMPLATE = "<read>{}</read>&#13;"
READNONE = "<read/>&#13;"
TRANSITIONTEMPLATE = """
        <transition>&#13;
			<from>{}</from>&#13;
			<to>{}</to>&#13;
			{}
		</transition>&#13;
"""

def transitionToJFLAP(states, transitions):
	stateXML = ""
	for i in range(len(states)):
		if i == 0:
			TAGS = INITIAL
			if states[i][3] == 'A':
				TAGS = TAGS + "\n" + FINAL
			stateXML = stateXML + STATETEMPLATE.format(i, 200 + (200 * i), 200, TAGS)
		else:
			TAGS = ""
			if states[i][3] == 'A':
				TAGS = TAGS + "\n" + FINAL
			stateXML = stateXML + STATETEMPLATE.format(i, 200 + (200 * i), 200, TAGS)
	
	transitionXML = ""
	for i in range(len(transitions)):
		s = transitions[i]
		for j in range(len(s)):
			if j == 2:
				for to in s[j]:
					transitionXML = transitionXML + TRANSITIONTEMPLATE.format(i, to, READNONE)
			else:
				for to in s[j]:
					transitionXML = transitionXML + TRANSITIONTEMPLATE.format(i, to, READTEMPLATE.format(j))

	return JFLAPTEMPLATE.format(stateXML, transitionXML)


def main():
	pass

if __name__ == '__main__':
	main()