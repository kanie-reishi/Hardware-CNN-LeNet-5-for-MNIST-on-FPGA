module line_buffer_shift(
    input wire clk,
    input [7:0] data_in, // Input data stream
    input wire valid_in, // Input data valid signal
    // Output: pixel in each collumn of 5x5 window
    output reg [7:0] dout_line1, // Line buffer output 1
    output reg [7:0] dout_line2, // Line buffer output 2
    output reg [7:0] dout_line3, // Line buffer output 3
    output reg [7:0] dout_line4, // Line buffer output 4

    output reg valid_out // Output data valid signal

);
    // 1. Parameters
    parameter IMG_WIDTH = 32 ; // Width in MNIST
    localparam LB_DEPTH = 32 ; // Depth of line buffer

    // 2. Line buffer storage
    reg [7:0] line_buffer0 [0:LB_DEPTH-1];
    reg [7:0] line_buffer1 [0:LB_DEPTH-1];
    reg [7:0] line_buffer2 [0:LB_DEPTH-1];
    reg [7:0] line_buffer3 [0:LB_DEPTH-1];

    // Loop variable
    integer i;

    // 3. Control logic (coordinate counters)
    reg [4:0] x_cnt; // X coordinate counter
    reg [4:0] y_cnt; // Y coordinate counter

    // 4. Combinational logic to shift line buffers and update sliding window
    always @(posedge clk) begin
            if(valid_in) begin

            // A. Update coordinate counters

            if(x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                if(y_cnt == IMG_WIDTH - 1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end

            // B. Shift line buffers and output data
            dout_line4 <= line_buffer3[LB_DEPTH-1];
            line_buffer3[0] <= dout_line3;
            for(i = 1; i < LB_DEPTH; i = i + 1) begin
                line_buffer3[i] <= line_buffer3[i-1];
            end
            dout_line3 <= line_buffer2[LB_DEPTH-1];
            line_buffer2[0] <= dout_line2;
            for(i = 1; i < LB_DEPTH; i = i + 1) begin
                line_buffer2[i] <= line_buffer2[i-1];
            end
            dout_line2 <= line_buffer1[LB_DEPTH-1];
            line_buffer1[0] <= dout_line1;
            for(i = 1; i < LB_DEPTH; i = i + 1) begin
                line_buffer1[i] <= line_buffer1[i-1];
            end
            dout_line1 <= line_buffer0[LB_DEPTH-1];
            line_buffer0[0] <= data_in;
            for(i = 1; i < LB_DEPTH; i = i + 1) begin
                line_buffer0[i] <= line_buffer0[i-1];
            end
            end
    end
    // 5. Valid logic
    assign valid_out = (x_cnt >= 4 && y_cnt >= 4) && valid_in;
endmodule