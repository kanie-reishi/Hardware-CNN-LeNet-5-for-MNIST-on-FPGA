module conv_pe_5x5_c1 #(
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter OUT_WIDTH = 24 // 8bit * 8bit * 25 elements
)(
    input clk,
    input rst_n,
    input en,

    // 25 pixels from window array
    input [DATA_WIDTH-1:0] p00, p01, p02, p03, p04, p05,
    input [DATA_WIDTH-1:0] p10, p11, p12, p13, p14,
    input [DATA_WIDTH-1:0] p20, p21, p22, p23, p24,
    input [DATA_WIDTH-1:0] p30, p31, p32, p33, p34,
    input [DATA_WIDTH-1:0] p40, p41, p42, p43, p44,
    // 25 Weights
    input signed [WEIGHT_WIDTH-1:0] w00, w01, w02, w03, w04, w05,
    input signed [WEIGHT_WIDTH-1:0] w10, w11, w12, w13, w14,
    input signed [WEIGHT_WIDTH-1:0] w20, w21, w22, w23, w24,
    input signed [WEIGHT_WIDTH-1:0] w30, w31, w32, w33, w34,
    input signed [WEIGHT_WIDTH-1:0] w40, w41, w42, w43, w44,

    input signed [OUT_WIDTH-1:0] bias,

    output reg signed [OUT_WIDTH-1:0] conv_out,
    output reg valid_out
);
   // --- PIPELINE CONTROL ---
    // Dùng thanh ghi dịch 3 bit để quản lý độ trễ (Latency = 3)
    // bit[0]: valid của stage 1, bit[1]: stage 2, bit[2]: stage 3 (output)
    reg [2:0] valid_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
        end else begin       
        valid_pipe <= {valid_pipe[1:0], en}; // Shift Left
        valid_out <= valid_pipe[2];
        end
    end


    // --- STAGE 1: MULTIPLICATION (Nhân song song 25 cặp) ---
    // Input vào: Clock T -> Output ra: Clock T+1
    reg signed [16:0] mult [0:24]; // 8bit * 8bit = 16bit

    always @(posedge clk) begin
        // Không cần if(en), cứ nhân liên tục để tối ưu logic
        // Phép nhân signed và unsigned nên ta phải ép kiểu cho pixel
        mult[0]  <= $signed({1'b0, p00}) * w00; mult[1]  <= $signed({1'b0, p01}) * w01; mult[2]  <= $signed({1'b0, p02}) * w02; mult[3]  <= $signed({1'b0, p03}) * w03; mult[4]  <= $signed({1'b0, p04}) * w04;
        mult[5]  <= $signed({1'b0, p10}) * w10; mult[6]  <= $signed({1'b0, p11}) * w11; mult[7]  <= $signed({1'b0, p12}) * w12; mult[8]  <= $signed({1'b0, p13}) * w13; mult[9]  <= $signed({1'b0, p14}) * w14;
        mult[10] <= $signed({1'b0, p20}) * w20; mult[11] <= $signed({1'b0, p21}) * w21; mult[12] <= $signed({1'b0, p22}) * w22; mult[13] <= $signed({1'b0, p23}) * w23; mult[14] <= $signed({1'b0, p24}) * w24;
        mult[15] <= $signed({1'b0, p30}) * w30; mult[16] <= $signed({1'b0, p31}) * w31; mult[17] <= $signed({1'b0, p32}) * w32; mult[18] <= $signed({1'b0, p33}) * w33; mult[19] <= $signed({1'b0, p34}) * w34;
        mult[20] <= $signed({1'b0, p40}) * w40; mult[21] <= $signed({1'b0, p41}) * w41; mult[22] <= $signed({1'b0, p42}) * w42; mult[23] <= $signed({1'b0, p43}) * w43; mult[24] <= $signed({1'b0, p44}) * w44;
    end


    // --- STAGE 2: PARTIAL SUM (Cây cộng phần 1) ---
    // Thay vì cộng hết 25 số, ta chia thành 5 nhóm nhỏ để giảm delay
    // Input vào: Clock T+1 -> Output ra: Clock T+2
    reg signed [20:0] sum_part [0:4];

    always @(posedge clk) begin
        // Gom 5 số thành 1 tổng (Adder Tree level 1)
        sum_part[0] <= mult[0]  + mult[1]  + mult[2]  + mult[3]  + mult[4];
        sum_part[1] <= mult[5]  + mult[6]  + mult[7]  + mult[8]  + mult[9];
        sum_part[2] <= mult[10] + mult[11] + mult[12] + mult[13] + mult[14];
        sum_part[3] <= mult[15] + mult[16] + mult[17] + mult[18] + mult[19];
        sum_part[4] <= mult[20] + mult[21] + mult[22] + mult[23] + mult[24];
    end


    // --- STAGE 3: FINAL SUM & BIAS (Cây cộng phần 2) ---
    // Cộng nốt 5 tổng con và Bias
    // Input vào: Clock T+2 -> Output ra: Clock T+3
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_out <= 0;
        end else begin
            // Adder Tree level 2 + Bias
            conv_out <= sum_part[0] + sum_part[1] + sum_part[2] + 
                        sum_part[3] + sum_part[4] + bias;
        end
    end
endmodule