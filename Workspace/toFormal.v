
//verilog NFA template

module Automaton1(clk, in, reset, out);

input clk, in, reset;
output out;

reg [1:0] state;
wire out;
assign out = state[1];

always @(posedge clk or posedge reset)
	begin
		if (reset)
			state = 2'b01;
		else
			if (in)
				case (state)
					
					2'b00:
						state <= 2'b00;
					2'b01:
						state <= 2'b10;
					2'b10:
						state <= 2'b10;
					2'b11:
						state <= 2'b10;
				endcase
			else
				case (state)
					
					2'b00:
						state <= 2'b00;
					2'b01:
						state <= 2'b01;
					2'b10:
						state <= 2'b01;
					2'b11:
						state <= 2'b01;
				endcase
	 end

endmodule

//verilog NFA template

module Automaton2(clk, in, reset, out);

input clk, in, reset;
output out;

reg [1:0] state;
wire out;
assign out = state[1];

always @(posedge clk or posedge reset)
	begin
		if (reset)
			state = 2'b01;
		else
			if (in)
				case (state)
					
					2'b00:
						state <= 2'b00;
					2'b01:
						state <= 2'b10;
					2'b10:
						state <= 2'b10;
					2'b11:
						state <= 2'b10;
				endcase
			else
				case (state)
					
					2'b00:
						state <= 2'b00;
					2'b01:
						state <= 2'b01;
					2'b10:
						state <= 2'b01;
					2'b11:
						state <= 2'b01;
				endcase
	 end

endmodule

module Equals(clock, inVal, res);
input clock;
input inVal;
input res;

wire out1, out2;

Automaton1 A1(clk, in, reset, out1);

Automaton2 A2(clk, in, reset, out2);

`ifdef FORMAL
	assert property (out1 == out2);
`endif

endmodule

