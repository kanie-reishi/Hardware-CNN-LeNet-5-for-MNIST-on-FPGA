`timescale 1ns/1ps

module lenet5_top #(
    parameter IN_PIXEL_WIDTH = 8,
    parameter OUT_LOGITS_WIDTH = 32,
    parameter FRONTEND_OUT_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,
    
    // --- GIAO TIẾP VỚI BÊN NGOÀI (CAMERA / MEMORY / TESTBENCH) ---
    input  wire valid_in,
    input  wire [IN_PIXEL_WIDTH-1:0] pixel_in, // Luồng 1 pixel / 1 clock
    
    // --- KẾT QUẢ DỰ ĐOÁN CUỐI CÙNG ---
    output wire valid_out,
    output wire [OUT_LOGITS_WIDTH*10-1:0] logits_out // 10 điểm số (32-bit mỗi số)
);

    // ==========================================
    // DÂY KẾT NỐI FRONTEND VÀ BACKEND
    // ==========================================
    // Dây nối chuẩn Streaming: 16 kênh song song, mỗi kênh 32-bit (Tổng 512-bit)
    wire s4_to_c5_valid;
    wire [FRONTEND_OUT_WIDTH*16-1:0] s4_to_c5_data; 

    // ==========================================
    // INSTANTIATE FRONTEND (C1 -> S2 -> C3 -> S4)
    // ==========================================
    lenet5_frontend_top #(
        .IN_WIDTH(IN_PIXEL_WIDTH),
        .OUT_WIDTH(FRONTEND_OUT_WIDTH),
        .OUT_C1_WIDTH(24), // Như bạn đã định nghĩa
        .WEIGHT_WIDTH(8)
    ) u_frontend (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .out_s4(s4_to_c5_data),
        .valid_out_s4(s4_to_c5_valid)
    );

    // ==========================================
    // INSTANTIATE BACKEND (C5 -> F6 -> Output)
    // ==========================================
    lenet5_backend_top #(
        .IN_WIDTH(FRONTEND_OUT_WIDTH),
        .W_WIDTH(8),
        .OUT_WIDTH(OUT_LOGITS_WIDTH)
    ) u_backend (
        .clk(clk),
        .rst_n(rst_n),
        .s4_valid_out(s4_to_c5_valid),
        .s4_flat_data(s4_to_c5_data), 
        .top_valid_out(valid_out),
        .top_logits_out(logits_out)
    );

endmodule