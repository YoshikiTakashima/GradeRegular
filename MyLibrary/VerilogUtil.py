from tarjan.tc import tc

VERILOGTEMPLATE = """
//verilog NFA template

module statem(clk, in, reset, out);

input clk, in, reset;
output out;

reg [{}:0] state;
wire out;
assign out = {};

always @(posedge clk or posedge reset)
	begin
		if (reset) {{
			state = {};
		}} else {{
			if (in) {{
				case (state)
					{}
				endcase
			}} else {{
				case (state)
					{}
				endcase
			}}
		}}
	 end

endmodule
"""
CASEELEMENTTEMPLATE = """
					{0}'b{1}:
						state <= {0}'b{2};"""


def epsionTransitionDestList(transitions):
	singular = {}
	for i in range(len(transitions)):
		t = transitions[i]
		singular[i] = t[2]
	
	epsilonTransitions = tc(singular)
	result = []
	for i in range(len(epsilonTransitions)):
		current = epsilonTransitions[i]
		bitStr = ''
		for j in range(len(epsilonTransitions)):
			if j in current:
				bitStr = '1' + bitStr
			else:
				bitStr = '0' + bitStr
		result.append(bitStr)
	return result

def zeros(numStates):
	txt =''
	for i in range(numStates):
		txt = txt + '0'
	return txt

def makeAcceptText(states):
	txt = ""
	acceptList = []
	for i in range(len(states)):
		if states[i][3] == 'A':
			acceptList.append(i)
	
	if len(acceptList) > 0:
		txt = "state[{}]".format(acceptList[0])
		for i in range(1, len(acceptList)):
			txt = txt + " | state[{}]".format(acceptList[0])
	else:
		txt = "1'b0"
	
	return txt

def makePerStateTransitionStr(inVal, index, transitions):
	destSet = transitions[index][inVal]
	# print(transitions[index])
	# print(destSet)
	
	result = []
	for i in range(len(transitions)):
		if i in destSet:
			result.append('1')
		else:
			result.append('0')

	return "".join(result)

def bitOr(bStr1, bStr2):
	result = ''

	for i in range(min(len(bStr1), len(bStr2))):
		if bStr1[i] == '1' or bStr2[i] == '1':
			result += '1'
		else:
			result += '0'
	return result

def makeTransitionCaseText(inVal, transitions, epsilon):
	txt = ''
	for i in range(2 ** len(transitions)):
		current = "{0:b}".format(i)
		while len(current) < len(epsilon):
			current = '0' + current
			
		nextStr = zeros(len(current))
		for j in range(len(current)):
			if current[j] == '1':
				destsFromState = makePerStateTransitionStr(inVal, len(current) - 1 - j, transitions)
				nextStr = bitOr(nextStr, destsFromState)
		
		for k in range(len(nextStr)):
			if nextStr[len(nextStr) - 1 - k] == '1':
				orAdd = epsilon[len(nextStr) - 1 - k]
				nextStr = bitOr(nextStr, orAdd[::-1])

		txt = txt + CASEELEMENTTEMPLATE.format(len(current), current, nextStr[::-1])
	return txt


def transitionToVerilog(states, transitions):
	numStates = len(states)
	acceptText = makeAcceptText(states)
	epsilonDests = epsionTransitionDestList(transitions)

	caseText1 =  makeTransitionCaseText(1, transitions, epsilonDests)
	caseText0 =  makeTransitionCaseText(0, transitions, epsilonDests)
	
	init = epsilonDests[0] 
	initList = list(init)
	initList[-1] = '1'
	init = "".join(list(initList))
	zeroText = "{}'b".format(len(transitions)) + init
	 
	return VERILOGTEMPLATE.format(numStates - 1, acceptText, zeroText, caseText1, caseText0)

def main():
	# Below is Sipser Example 1.38
	exampleStates = [
		(0, 0, 0, 'NA'),
		(0, 0, 0, 'NA'),
		(0, 0, 0, 'NA'),
		(0, 0, 0, 'A')
	]
	exampleTransitions = [
		([0], [0, 1], []),
		([2], [], [2]),
		([], [3], []),
		([3], [3], []), 
	]
	print(transitionToVerilog(exampleStates, exampleTransitions))

if __name__ == '__main__':
	main()