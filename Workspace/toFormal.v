//verilog NFA template

module Automaton1(clk, in, reset, out);

input clk, in, reset;
output out;

reg [1:0] state;
wire out;
assign out = state[1];

always @(posedge clk or posedge reset)
	begin
		if (reset) {
			state = 2'b01;
		} else {
			if (in) {
				case (state)
					
					2'b00:
						state = 2'b00;
					2'b01:
						state = 2'b10;
					2'b10:
						state = 2'b10;
					2'b11:
						state = 2'b10;
				endcase
			} else {
				case (state)
					
					2'b00:
						state = 2'b00;
					2'b01:
						state = 2'b01;
					2'b10:
						state = 2'b01;
					2'b11:
						state = 2'b01;
				endcase
			}
		}
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
		if (reset) {
			state = 2'b01;
		} else {
			if (in) {
				case (state)
					
					2'b00:
						state = 2'b00;
					2'b01:
						state = 2'b11;
					2'b10:
						state = 2'b00;
					2'b11:
						state = 2'b11;
				endcase
			} else {
				case (state)
					
					2'b00:
						state = 2'b00;
					2'b01:
						state = 2'b01;
					2'b10:
						state = 2'b00;
					2'b11:
						state = 2'b01;
				endcase
			}
		}
	 end

endmodule


module Equals(clock, inVal, res);
input clock, inVal, res;
wire clock, inVal, res;

wire out1, out2;

Automaton1 A1 (
	.clk	(closk),
	.in		(inVal),
	.reset	(res)
	.out	(out1)
);

Automaton2 A2 (
	.clk	(clock),
	.in		(inVal),
	.reset	(res)
	.out	(out2)
);

always @(posedge clock) begin
	if(~reset)
		assert(out1 == out2);
	end
endmodule
