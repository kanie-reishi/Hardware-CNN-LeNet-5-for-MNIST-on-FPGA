module pooling_layer_s2 (
    input clk,
    input rst_n,

    // Input feature map pixel stream
    input signed [23:0] feat_map_in_1,
    input signed [23:0] feat_map_in_2,
    input signed [23:0] feat_map_in_3,
    input signed [23:0] feat_map_in_4,
    input signed [23:0] feat_map_in_5,
    input signed [23:0] feat_map_in_6,
    input valid_in,

    // Output pooled feature map pixel stream
    output signed [23:0] pooled_out_1,
    output signed [23:0] pooled_out_2,
    output signed [23:0] pooled_out_3,
    output signed [23:0] pooled_out_4,
    output signed [23:0] pooled_out_5,
    output signed [23:0] pooled_out_6,
    output wire valid_out
);
    // Wire for 6 valid outputs from pooling units
    wire valid_out_1, valid_out_2, valid_out_3, valid_out_4, valid_out_5, valid_out_6;
    // Instantiate 6 Pooling Units (2x2, Stride 2)
    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu1 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_1),
        .valid_in(valid_in),
        .data_out(pooled_out_1),
        .valid_out(valid_out_1)
    );

    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu2 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_2),
        .valid_in(valid_in),
        .data_out(pooled_out_2),
        .valid_out(valid_out_2)
    );

    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu3 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_3),
        .valid_in(valid_in),
        .data_out(pooled_out_3),
        .valid_out(valid_out_3)
    );

    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu4 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_4),
        .valid_in(valid_in),
        .data_out(pooled_out_4),
        .valid_out(valid_out_4)
    );

    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu5 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_5),
        .valid_in(valid_in),
        .data_out(pooled_out_5),
        .valid_out(valid_out_5)
    );

    pooling_unit_2x2 #(
        .WIDTH(24),
        .IMG_WIDTH(28)
    ) pu6 (
        .clk(clk),
        .rst_n(rst_n),
        .feat_map_in(feat_map_in_6),
        .valid_in(valid_in),
        .data_out(pooled_out_6),
        .valid_out(valid_out_6)
    );

    // Combine valid outputs - one valid when all pooling units are valid
    assign valid_out = valid_out_1 ;
endmodule