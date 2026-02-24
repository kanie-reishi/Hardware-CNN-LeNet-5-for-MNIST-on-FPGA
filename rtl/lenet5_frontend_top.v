module lenet5_frontend_top #(
    parameter IN_WIDTH = 8,
    parameter OUT_WIDTH = 32,
    parameter OUT_C1_WIDTH = 24,
    parameter WEIGHT_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire [IN_WIDTH - 1:0] pixel_in, // 8-bit pixel stream
    input wire valid_in,
    output [OUT_WIDTH*16 - 1:0] out_s4, // 16 feature maps, mỗi map 32 bits
    output wire valid_out_s4
);
    // --- 1. CONVOLUTION LAYER C1 ---
    wire signed [OUT_C1_WIDTH-1:0] c1_out_0, c1_out_1, c1_out_2, c1_out_3, c1_out_4, c1_out_5;
    wire valid_c1_out;
    conv_layer_c1 c1 (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .fmap_out_0(c1_out_0), .fmap_out_1(c1_out_1), .fmap_out_2(c1_out_2),
        .fmap_out_3(c1_out_3), .fmap_out_4(c1_out_4), .fmap_out_5(c1_out_5),
        .valid_out(valid_c1_out)
    );
    // --- 2. POOLING LAYER S2 ---
    wire signed [OUT_C1_WIDTH-1:0] s2_out_0, s2_out_1, s2_out_2, s2_out_3, s2_out_4, s2_out_5;
    wire valid_s2_out;
    pooling_layer_s2 s2 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in_1(c1_out_0), .feat_map_in_2(c1_out_1), .feat_map_in_3(c1_out_2),
        .feat_map_in_4(c1_out_3), .feat_map_in_5(c1_out_4), .feat_map_in_6(c1_out_5),
        .valid_in(valid_c1_out),
        .pooled_out_1(s2_out_0), .pooled_out_2(s2_out_1), .pooled_out_3(s2_out_2),
        .pooled_out_4(s2_out_3), .pooled_out_5(s2_out_4), .pooled_out_6(s2_out_5),
        .valid_out(valid_s2_out)
    );
    // --- 3. CONVOLUTION LAYER C3 ---
    wire signed [OUT_WIDTH-1:0] c3_out_0, c3_out_1, c3_out_2, c3_out_3, c3_out_4, c3_out_5, c3_out_6;
    wire signed [OUT_WIDTH-1:0] c3_out_7, c3_out_8, c3_out_9, c3_out_10, c3_out_11, c3_out_12;
    wire signed [OUT_WIDTH-1:0] c3_out_13, c3_out_14, c3_out_15;
    wire valid_c3_out;
    conv_layer_c3 #(
        .IN_WIDTH(OUT_C1_WIDTH),
        .W_WIDTH(WEIGHT_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) c3 (
        .clk(clk),
        .rst_n(rst_n),
        .fmap_in_0(s2_out_0), .fmap_in_1(s2_out_1), .fmap_in_2(s2_out_2),
        .fmap_in_3(s2_out_3), .fmap_in_4(s2_out_4), .fmap_in_5(s2_out_5),
        .valid_in(valid_s2_out),
        .fmap_out_0(c3_out_0), .fmap_out_1(c3_out_1), .fmap_out_2(c3_out_2),
        .fmap_out_3(c3_out_3), .fmap_out_4(c3_out_4), .fmap_out_5(c3_out_5),
        .fmap_out_6(c3_out_6), .fmap_out_7(c3_out_7), .fmap_out_8(c3_out_8),
        .fmap_out_9(c3_out_9), .fmap_out_10(c3_out_10), .fmap_out_11(c3_out_11),
        .fmap_out_12(c3_out_12), .fmap_out_13(c3_out_13), .fmap_out_14(c3_out_14),
        .fmap_out_15(c3_out_15),
        .valid_out(valid_c3_out)
    );
    // --- 4. POOLING LAYER S4 ---
    wire signed [OUT_WIDTH-1:0] s4_out_0, s4_out_1, s4_out_2, s4_out_3, s4_out_4, s4_out_5;
    wire signed [OUT_WIDTH-1:0] s4_out_6, s4_out_7, s4_out_8, s4_out_9, s4_out_10, s4_out_11;
    wire signed [OUT_WIDTH-1:0] s4_out_12, s4_out_13, s4_out_14, s4_out_15;
    wire valid_s4_out;
    pooling_layer_s4 s4 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in_0(c3_out_0), .feat_map_in_1(c3_out_1), .feat_map_in_2(c3_out_2),
        .feat_map_in_3(c3_out_3), .feat_map_in_4(c3_out_4), .feat_map_in_5(c3_out_5),
        .feat_map_in_6(c3_out_6), .feat_map_in_7(c3_out_7), .feat_map_in_8(c3_out_8),
        .feat_map_in_9(c3_out_9), .feat_map_in_10(c3_out_10), .feat_map_in_11(c3_out_11),
        .feat_map_in_12(c3_out_12), .feat_map_in_13(c3_out_13), .feat_map_in_14(c3_out_14),
        .feat_map_in_15(c3_out_15),
        .valid_in(valid_c3_out),
        .pooled_out_0(s4_out_0), .pooled_out_1(s4_out_1), .pooled_out_2(s4_out_2),
        .pooled_out_3(s4_out_3), .pooled_out_4(s4_out_4), .pooled_out_5(s4_out_5),
        .pooled_out_6(s4_out_6), .pooled_out_7(s4_out_7), .pooled_out_8(s4_out_8),
        .pooled_out_9(s4_out_9), .pooled_out_10(s4_out_10), .pooled_out_11(s4_out_11),
        .pooled_out_12(s4_out_12), .pooled_out_13(s4_out_13), .pooled_out_14(s4_out_14),
        .pooled_out_15(s4_out_15),
        .valid_out(valid_s4_out)
    );
    // --- 5. OUTPUT ASSIGNMENT ---
    // Flatten 16 pooled outputs from S4 into a single bus
    assign out_s4 = {s4_out_0, s4_out_1, s4_out_2, s4_out_3, s4_out_4, s4_out_5, s4_out_6, s4_out_7,
                     s4_out_8, s4_out_9, s4_out_10, s4_out_11, s4_out_12, s4_out_13, s4_out_14, s4_out_15};
    assign valid_out_s4 = valid_s4_out;
endmodule