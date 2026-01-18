
module conv_layer_c1 (
    input clk,
    input rst_n,

    // Input image pixel stream
    input [7:0] pixel_in,
    input valid_in,

    // Output feature map pixel stream
    // 6 output channels
    // Flattened output
    output reg signed [23:0] fmap_out_0,
    output reg signed [23:0] fmap_out_1,
    output reg signed [23:0] fmap_out_2,
    output reg signed [23:0] fmap_out_3,
    output reg signed [23:0] fmap_out_4,
    output reg signed [23:0] fmap_out_5,

    output wire valid_out
);
    // === LINE BUFFER & WINDOW ARRAY for 5x5 ===
    wire [7:0] w_p00, w_p01, w_p02, w_p03, w_p04;
    wire [7:0] w_p10, w_p11, w_p12, w_p13, w_p14;
    wire [7:0] w_p20, w_p21, w_p22, w_p23, w_p24;
    wire [7:0] w_p30, w_p31, w_p32, w_p33, w_p34;
    wire [7:0] w_p40, w_p41, w_p42, w_p43, w_p44;
    wire window_valid;

    line_buffer_window_5x5 #(
        .IMG_WIDTH(32)
    ) lb_win_5x5 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(pixel_in),
        .valid_in(valid_in),
        .p00(w_p00), .p01(w_p01), .p02(w_p02), .p03(w_p03), .p04(w_p04),
        .p10(w_p10), .p11(w_p11), .p12(w_p12), .p13(w_p13), .p14(w_p14),
        .p20(w_p20), .p21(w_p21), .p22(w_p22), .p23(w_p23), .p24(w_p24),
        .p30(w_p30), .p31(w_p31), .p32(w_p32), .p33(w_p33), .p34(w_p34),
        .p40(w_p40), .p41(w_p41), .p42(w_p42), .p43(w_p43), .p44(w_p44),
        .valid_out(window_valid)
    );

    // === WEIGHTS STORAGE ===
    // Weights for 6 filters (5x5 each)
    // 150 Weights (6 fileters * 25 weights)
    reg signed [7:0] KERNEL [0:149];
    reg signed [23:0] BIAS [0:5];

    initial begin
        // Load Hex files for weights and biases
        // Or hardcode them here
        $readmemh("c1_weight.hex", KERNEL);
        // Example: $readmemh("c1_bias.hex", BIAS);
        // Make biases zero for testing
        BIAS[0] = 20'sd0;
        BIAS[1] = 20'sd0;
        BIAS[2] = 20'sd0;
        BIAS[3] = 20'sd0;
        BIAS[4] = 20'sd0;
        BIAS[5] = 20'sd0;
    end
    // === CONVOLUTION PEs for 6 output channels ===
    // Parallel convolution processing elements
    wire signed [23:0] pe_results [0:5];
    wire [5:0] pe_valid_out;

    genvar i;
    generate
        for(i = 0; i < 6; i = i + 1) begin : CONV_ENGINES
            // Offset for each kernel
            localparam KERNEL_OFFSET = i * 25;
            conv_pe_c1 #(
                .DATA_WIDTH(8),
                .WEIGHT_WIDTH(8),
                .OUT_WIDTH(24)
            ) conv_pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                .en(window_valid),

                // 25 pixels from window array
                .p00(w_p00), .p01(w_p01), .p02(w_p02), .p03(w_p03), .p04(w_p04),
                .p10(w_p10), .p11(w_p11), .p12(w_p12), .p13(w_p13), .p14(w_p14),
                .p20(w_p20), .p21(w_p21), .p22(w_p22), .p23(w_p23), .p24(w_p24),
                .p30(w_p30), .p31(w_p31), .p32(w_p32), .p33(w_p33), .p34(w_p34),
                .p40(w_p40), .p41(w_p41), .p42(w_p42), .p43(w_p43), .p44(w_p44),

                // 25 Weights
                .w00(KERNEL[KERNEL_OFFSET + 0]), .w01(KERNEL[KERNEL_OFFSET + 1]), .w02(KERNEL[KERNEL_OFFSET + 2]), .w03(KERNEL[KERNEL_OFFSET + 3]), .w04(KERNEL[KERNEL_OFFSET + 4]),
                .w10(KERNEL[KERNEL_OFFSET + 5]), .w11(KERNEL[KERNEL_OFFSET + 6]), .w12(KERNEL[KERNEL_OFFSET + 7]), .w13(KERNEL[KERNEL_OFFSET + 8]), .w14(KERNEL[KERNEL_OFFSET + 9]),
                .w20(KERNEL[KERNEL_OFFSET + 10]), .w21(KERNEL[KERNEL_OFFSET + 11]), .w22(KERNEL[KERNEL_OFFSET + 12]), .w23(KERNEL[KERNEL_OFFSET + 13]), .w24(KERNEL[KERNEL_OFFSET + 14]),
                .w30(KERNEL[KERNEL_OFFSET + 15]), .w31(KERNEL[KERNEL_OFFSET + 16]), .w32(KERNEL[KERNEL_OFFSET + 17]), .w33(KERNEL[KERNEL_OFFSET + 18]), .w34(KERNEL[KERNEL_OFFSET + 19]),
                .w40(KERNEL[KERNEL_OFFSET + 20]), .w41(KERNEL[KERNEL_OFFSET + 21]), .w42(KERNEL[KERNEL_OFFSET + 22]), .w43(KERNEL[KERNEL_OFFSET + 23]), .w44(KERNEL[KERNEL_OFFSET + 24]),
                .bias(BIAS[i]),
                // Outputs
                .conv_out(pe_results[i]),
                .valid_out(pe_valid_out[i])
            );
        end
    endgenerate
    // === OUTPUT ASSIGNMENT ===
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            fmap_out_0 <= 0;
            fmap_out_1 <= 0;
            fmap_out_2 <= 0;
            fmap_out_3 <= 0;
            fmap_out_4 <= 0;
            fmap_out_5 <= 0;
        end else begin
            fmap_out_0 <= pe_results[0];
            fmap_out_1 <= pe_results[1];
            fmap_out_2 <= pe_results[2];
            fmap_out_3 <= pe_results[3];
            fmap_out_4 <= pe_results[4];
            fmap_out_5 <= pe_results[5];
        end
    end
    // Valid output when any PE produces valid output
    assign valid_out = |pe_valid_out;
endmodule