module line_buffer_window_5x5 #(
    parameter IMG_WIDTH = 32
)(
    input clk,
    input rst_n,
    input [7:0] data_in,
    input valid_in,
    // 25 pixel output (Flattened)
    output wire [7:0] p00, p01, p02, p03, p04,
    output wire [7:0] p10, p11, p12, p13, p14,
    output wire [7:0] p20, p21, p22, p23, p24,
    output wire [7:0] p30, p31, p32, p33, p34,
    output wire [7:0] p40, p41, p42, p43, p44,
    output reg valid_out // Output valid signal
);
    // Intermediate wires for line buffer outputs
    wire [7:0] line1_out;
    wire [7:0] line2_out;
    wire [7:0] line3_out;
    wire [7:0] line4_out;
    wire lb_valid_out; // Valid signal from line buffer
    // Call line buffer shift module
    line_buffer_shift #(
        .IMG_WIDTH(IMG_WIDTH)
    ) lb_shift (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .valid_in(valid_in),
        .dout_line1(line1_out), // Dòng t-1
        .dout_line2(line2_out), // Dòng t-2
        .dout_line3(line3_out), // Dòng t-3
        .dout_line4(line4_out), // Dòng t-4
        .valid_out(lb_valid_out) 
    );
    // Call window array module
    window_array win_array (
        .clk(clk),
        .shift_en(valid_in),
        .rst_n(rst_n),
        .row_in_0(data_in),      // Dòng hiện tại (t)
        .row_in_1(line1_out),    // Dòng t-1
        .row_in_2(line2_out),    // Dòng t-2
        .row_in_3(line3_out),    // Dòng t-3
        .row_in_4(line4_out),    // Dòng t-4
        // 25 pixel outputs
        .w00(p00), .w01(p01), .w02(p02), .w03(p03), .w04(p04),
        .w10(p10), .w11(p11), .w12(p12), .w13(p13), .w14(p14),
        .w20(p20), .w21(p21), .w22(p22), .w23(p23), .w24(p24),
        .w30(p30), .w31(p31), .w32(p32), .w33(p33), .w34(p34),
        .w40(p40), .w41(p41), .w42(p42), .w43(p43), .w44(p44)
    );
    // Logic control valid_out, delay lb_valid_out by 1 cycle
    always @(posedge clk) begin
        if(!rst_n) begin
            valid_out <= 0;
        end else begin
            valid_out <= lb_valid_out;
        end
    end
endmodule