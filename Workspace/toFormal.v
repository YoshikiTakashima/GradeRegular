
//verilog NFA template

module Automaton1(input clk, in, reset, output out);

reg [1:0] state = 2'b01;
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

module Automaton2(input clk, in, reset, output out);

reg [2:0] state = 3'b001;
assign out = state[1];

always @(posedge clk or posedge reset)
	begin
		if (reset)
			state = 3'b001;
		else
			if (in)
				case (state)
					
					3'b000:
						state <= 3'b000;
					3'b001:
						state <= 3'b110;
					3'b010:
						state <= 3'b010;
					3'b011:
						state <= 3'b110;
					3'b100:
						state <= 3'b000;
					3'b101:
						state <= 3'b110;
					3'b110:
						state <= 3'b010;
					3'b111:
						state <= 3'b110;
				endcase
			else
				case (state)
					
					3'b000:
						state <= 3'b000;
					3'b001:
						state <= 3'b001;
					3'b010:
						state <= 3'b001;
					3'b011:
						state <= 3'b001;
					3'b100:
						state <= 3'b000;
					3'b101:
						state <= 3'b001;
					3'b110:
						state <= 3'b001;
					3'b111:
						state <= 3'b001;
				endcase
	 end

endmodule

module Equals(input clock, inVal, res);

wire out1, out2;

Automaton1 A1(.clk(clock), .in(inVal), .reset(res), .out(out1));

Automaton2 A2(.clk(clock), .in(inVal), .reset(res), .out(out2));

assert property (out1 == out2);

endmodule

