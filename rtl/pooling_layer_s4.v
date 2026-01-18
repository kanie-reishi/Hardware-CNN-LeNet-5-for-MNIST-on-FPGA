module pooling_layer_s4 #(
    parameter WIDTH = 32, // Input from C3 is 32 bits
    parameter IMG_WIDTH = 10 // Input feature map width
)(
    input clk,
    input rst_n,
    
    // Input : 16 feature map pixel streams from C3
    input signed [WIDTH-1:0] feat_map_in_0, feat_map_in_1, feat_map_in_2, feat_map_in_3,
    input signed [WIDTH-1:0] feat_map_in_4, feat_map_in_5, feat_map_in_6, feat_map_in_7,
    input signed [WIDTH-1:0] feat_map_in_8, feat_map_in_9, feat_map_in_10, feat_map_in_11,
    input signed [WIDTH-1:0] feat_map_in_12, feat_map_in_13, feat_map_in_14, feat_map_in_15,
    input valid_in,

    // Output : 16 pooled feature map pixel streams
    output reg signed [WIDTH-1:0] pooled_out_0, pooled_out_1, pooled_out_2, pooled_out_3,
    output reg signed [WIDTH-1:0] pooled_out_4, pooled_out_5, pooled_out_6, pooled_out_7,
    output reg signed [WIDTH-1:0] pooled_out_8, pooled_out_9, pooled_out_10, pooled_out_11,
    output reg signed [WIDTH-1:0] pooled_out_12, pooled_out_13, pooled_out_14, pooled_out_15,
    output wire valid_out
);
    //1. DECLAREMENTS
    // Wire inputs into an array for easier handling
    wire signed [WIDTH-1:0] feat_map_in_array [0:15];
    assign feat_map_in_array[0] = feat_map_in_0; assign feat_map_in_array[1] = feat_map_in_1;
    assign feat_map_in_array[2] = feat_map_in_2; assign feat_map_in_array[3] = feat_map_in_3;
    assign feat_map_in_array[4] = feat_map_in_4; assign feat_map_in_array[5] = feat_map_in_5;
    assign feat_map_in_array[6] = feat_map_in_6; assign feat_map_in_array[7] = feat_map_in_7;
    assign feat_map_in_array[8] = feat_map_in_8; assign feat_map_in_array[9] = feat_map_in_9;
    assign feat_map_in_array[10] = feat_map_in_10; assign feat_map_in_array[11] = feat_map_in_11;
    assign feat_map_in_array[12] = feat_map_in_12; assign feat_map_in_array[13] = feat_map_in_13;
    assign feat_map_in_array[14] = feat_map_in_14; assign feat_map_in_array[15] = feat_map_in_15;

    // Wires for outputs array from pooling units
    wire signed [WIDTH-1:0] pooled_out_array [0:15];
    wire [15:0] valid_out_array;

    //2. INSTANTIATE 16 POOLING UNITS (2x2, STRIDE 2)
    genvar i;
    generate
        for(i = 0;i < 16;i = i + 1) begin: POOL_S4_GEN
            pooling_unit_2x2 #(
                .WIDTH(WIDTH),
                .IMG_WIDTH(IMG_WIDTH)
            ) pu_s4 (
                .clk(clk),
                .rst_n(rst_n),
                .feat_map_in(feat_map_in_array[i]),
                .valid_in(valid_in),
                .data_out(pooled_out_array[i]),
                .valid_out(valid_out_array[i])
            );
        end
    endgenerate
    //3. ASSIGN OUTPUTS
    always @(*) begin
        pooled_out_0 = pooled_out_array[0];
        pooled_out_1 = pooled_out_array[1];
        pooled_out_2 = pooled_out_array[2];
        pooled_out_3 = pooled_out_array[3];
        pooled_out_4 = pooled_out_array[4];
        pooled_out_5 = pooled_out_array[5];
        pooled_out_6 = pooled_out_array[6];
        pooled_out_7 = pooled_out_array[7];
        pooled_out_8 = pooled_out_array[8];
        pooled_out_9 = pooled_out_array[9];
        pooled_out_10 = pooled_out_array[10];
        pooled_out_11 = pooled_out_array[11];
        pooled_out_12 = pooled_out_array[12];
        pooled_out_13 = pooled_out_array[13];
        pooled_out_14 = pooled_out_array[14];
        pooled_out_15 = pooled_out_array[15];
    end
    // Valid output when any pooling unit produces valid output because parallel
    assign valid_out = valid_out_array[0];
endmodule