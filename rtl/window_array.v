module window_array #(
    parameter DATA_WIDTH = 8
)(
    input clk,
    input shift_en,
    input rst_n,
    // 5 dòng dữ liệu đầu vào song song
    input [DATA_WIDTH-1:0] row_in_0, // Dòng hiện tại (t)
    input [DATA_WIDTH-1:0] row_in_1, // Dòng t-1
    input [DATA_WIDTH-1:0] row_in_2, // Dòng t-2
    input [DATA_WIDTH-1:0] row_in_3, // Dòng t-3
    input [DATA_WIDTH-1:0] row_in_4, // Dòng t-4
    
    // Output 25 pixel (Flattened)
    output reg [DATA_WIDTH-1:0] w00, w01, w02, w03, w04, // Hàng t-4
    output reg [DATA_WIDTH-1:0] w10, w11, w12, w13, w14, // Hàng t-3
    output reg [DATA_WIDTH-1:0] w20, w21, w22, w23, w24, // Hàng t-2
    output reg [DATA_WIDTH-1:0] w30, w31, w32, w33, w34, // Hàng t-1
    output reg [DATA_WIDTH-1:0] w40, w41, w42, w43, w44  // Hàng t (Hiện tại)
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset tất cả các thanh ghi cửa sổ
            w00 <= 0; w01 <= 0; w02 <= 0; w03 <= 0; w04 <= 0;
            w10 <= 0; w11 <= 0; w12 <= 0; w13 <= 0; w14 <= 0;
            w20 <= 0; w21 <= 0; w22 <= 0; w23 <= 0; w24 <= 0;
            w30 <= 0; w31 <= 0; w32 <= 0; w33 <= 0; w34 <= 0;
            w40 <= 0; w41 <= 0; w42 <= 0; w43 <= 0; w44 <= 0;
        end else if (shift_en) begin
            // Hàng 4 (Dòng hiện tại)
            w44 <= row_in_0; w43 <= w44; w42 <= w43; w41 <= w42; w40 <= w41;
            
            // Hàng 3 (Dòng t-1)
            w34 <= row_in_1; w33 <= w34; w32 <= w33; w31 <= w32; w30 <= w31;
            
            // Hàng 2 (Dòng t-2)
            w24 <= row_in_2; w23 <= w24; w22 <= w23; w21 <= w22; w20 <= w21;
            
            // Hàng 1 (Dòng t-3)
            w14 <= row_in_3; w13 <= w14; w12 <= w13; w11 <= w12; w10 <= w11;
            
            // Hàng 0 (Dòng t-4)
            w04 <= row_in_4; w03 <= w04; w02 <= w03; w01 <= w02; w00 <= w01;
        end
    end
endmodule