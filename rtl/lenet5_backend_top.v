`timescale 1ns/1ps

module lenet5_backend_top #(
    parameter IN_WIDTH = 32,
    parameter W_WIDTH = 8,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    
    // --- GIAO TIẾP VỚI TẦNG S4 ---
    input wire s4_valid_out,
    // Bus chứa 16 feature maps từ S4, mỗi map 25 pixels (5x5) và mỗi pixel 32 bits
    input wire [IN_WIDTH*16-1:0] s4_flat_data, 
    
    // --- GIAO TIẾP OUTPUT ---
    output wire top_valid_out,
    // Bus chứa 10 logits (10 * 32 = 320 bits)
    output wire [OUT_WIDTH*10-1:0] top_logits_out
);

    // ==========================================
    // KHAI BÁO DÂY KẾT NỐI (INTERCONNECT WIRES)
    // ==========================================
    
    // Dây từ C5 sang F6
    wire c5_to_f6_valid;
    wire [OUT_WIDTH*120-1:0] c5_to_f6_data; // 120 outputs
    
    // Dây từ F6 sang Output Layer
    wire f6_to_out_valid;
    wire [OUT_WIDTH*84-1:0] f6_to_out_data; // 84 outputs

    // ==========================================
    // INSTANTIATE CÁC MODULE CON
    // ==========================================

    // 1. Tầng C5 (Convolution / Fully Connected 1)
    conv_layer_c5 #(
        .IN_WIDTH(IN_WIDTH),
        .W_WIDTH(W_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .NUM_PE(60)
    ) u_layer_c5 (
        .clk(clk),
        .rst_n(rst_n),
        .flat_in_data(s4_flat_data),
        .in_valid(s4_valid_out),
        .flat_out_data(c5_to_f6_data),
        .valid_out(c5_to_f6_valid)
    );

    // 2. Tầng F6 (Fully Connected 2)
    fc_layer_f6 #(
        .IN_WIDTH(IN_WIDTH),
        .W_WIDTH(W_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .NUM_IN(120),
        .NUM_OUT(84)
        // .SHIFT_AMT(6)
    ) u_layer_f6 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(c5_to_f6_valid),
        .in_data(c5_to_f6_data),
        .out_data(f6_to_out_data),
        .valid_out(f6_to_out_valid)
    );

    // 3. Tầng Output (Fully Connected 3 - Logits)
    output_layer #(
        .IN_WIDTH(IN_WIDTH),
        .W_WIDTH(W_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .NUM_IN(84),
        .NUM_OUT(10)
        // .SHIFT_AMT(5)
    ) u_layer_out (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(f6_to_out_valid),
        .in_data(f6_to_out_data),
        .out_data(top_logits_out),
        .valid_out(top_valid_out)
    );

endmodule