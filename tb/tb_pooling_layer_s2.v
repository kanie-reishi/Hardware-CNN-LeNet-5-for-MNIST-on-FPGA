`timescale 1ns/1ps

module tb_pooling_layer_s2;

    // --- 1. KHAI BÁO TÍN HIỆU ---
    reg clk;
    reg rst_n;
    reg valid_in;
    
    // Input 6 kênh (24-bit)
    reg signed [23:0] fm_in_1, fm_in_2, fm_in_3, fm_in_4, fm_in_5, fm_in_6;
    
    // Output 6 kênh
    wire signed [23:0] pool_out_1, pool_out_2, pool_out_3, pool_out_4, pool_out_5, pool_out_6;
    wire valid_out;

    // Biến hỗ trợ đọc/ghi file
    integer file_in, file_out;
    integer scan_res;
    integer i;

    // --- 2. INSTANTIATE DUT ---
    pooling_layer_s2 u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .feat_map_in_1(fm_in_1), .feat_map_in_2(fm_in_2),
        .feat_map_in_3(fm_in_3), .feat_map_in_4(fm_in_4),
        .feat_map_in_5(fm_in_5), .feat_map_in_6(fm_in_6),
        
        .pooled_out_1(pool_out_1), .pooled_out_2(pool_out_2),
        .pooled_out_3(pool_out_3), .pooled_out_4(pool_out_4),
        .pooled_out_5(pool_out_5), .pooled_out_6(pool_out_6),
        .valid_out(valid_out)
    );

    // --- 3. CLOCK GENERATION ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    // --- 4. MAIN PROCESS ---
    initial begin
        // Mở file Input (Log của C1)
        file_in = $fopen("output_c1_log.txt", "r");
        if (file_in == 0) begin
            $display("ERROR: Cannot open output_c1_log.txt");
            $finish;
        end

        // Mở file Output (Log của S2)
        file_out = $fopen("output_s2_log.txt", "w");

        // Reset
        rst_n = 0;
        valid_in = 0;
        fm_in_1 = 0; fm_in_2 = 0; fm_in_3 = 0; fm_in_4 = 0; fm_in_5 = 0; fm_in_6 = 0;
        #100;
        rst_n = 1;
        #20;

        $display("STARTING POOLING LAYER SIMULATION...");

        // Loop đọc 784 dòng (28x28) từ file C1 Log
        // Lưu ý: Nếu file C1 có ít hơn 784 dòng hợp lệ, vòng lặp sẽ dừng sớm
        while (!$feof(file_in)) begin
            @(negedge clk);
            
            // Đọc 6 số nguyên từ 1 dòng
            scan_res = $fscanf(file_in, "%d %d %d %d %d %d\n", 
                               fm_in_1, fm_in_2, fm_in_3, fm_in_4, fm_in_5, fm_in_6);
            
            // Nếu đọc thành công đủ 6 số
            if (scan_res == 6) begin
                valid_in = 1;
            end else begin
                valid_in = 0;
            end
        end

        // Hết dữ liệu
        @(negedge clk);
        valid_in = 0;
        
        // Chờ xử lý nốt pipeline (khoảng vài chục clock là đủ cho Pooling)
        #500;
        
        $display("SIMULATION FINISHED.");
        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

    // --- 5. MONITOR & SAVE OUTPUT ---
    always @(posedge clk) begin
        if (valid_out) begin
            // Ghi kết quả ra file log S2
            $fwrite(file_out, "%d %d %d %d %d %d\n", 
                    pool_out_1, pool_out_2, pool_out_3, 
                    pool_out_4, pool_out_5, pool_out_6);
        end
    end

endmodule