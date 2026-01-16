`timescale 1ns/1ps

module tb_conv_layer_c3;

    // --- 1. PARAMETERS ---
    parameter IN_WIDTH = 24;  // Input từ S2
    parameter OUT_WIDTH = 32; // Output của C3 (32-bit để chứa số lớn)

    // --- 2. SIGNALS ---
    reg clk;
    reg rst_n;
    reg valid_in;

    // 6 Input Channels
    reg signed [IN_WIDTH-1:0] fmap_in_0, fmap_in_1, fmap_in_2, fmap_in_3, fmap_in_4, fmap_in_5;

    // 16 Output Channels
    wire signed [OUT_WIDTH-1:0] fmap_out_0,  fmap_out_1,  fmap_out_2,  fmap_out_3;
    wire signed [OUT_WIDTH-1:0] fmap_out_4,  fmap_out_5,  fmap_out_6,  fmap_out_7;
    wire signed [OUT_WIDTH-1:0] fmap_out_8,  fmap_out_9,  fmap_out_10, fmap_out_11;
    wire signed [OUT_WIDTH-1:0] fmap_out_12, fmap_out_13, fmap_out_14, fmap_out_15;
    
    wire valid_out;

    // File Handlers
    integer file_in, file_out;
    integer scan_res;

    // --- 3. INSTANTIATE DUT (Device Under Test) ---
    conv_layer_c3 #(
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        
        // Connect Inputs
        .fmap_in_0(fmap_in_0), .fmap_in_1(fmap_in_1), .fmap_in_2(fmap_in_2),
        .fmap_in_3(fmap_in_3), .fmap_in_4(fmap_in_4), .fmap_in_5(fmap_in_5),
        .valid_in(valid_in),

        // Connect Outputs
        .fmap_out_0(fmap_out_0),   .fmap_out_1(fmap_out_1),   .fmap_out_2(fmap_out_2),   .fmap_out_3(fmap_out_3),
        .fmap_out_4(fmap_out_4),   .fmap_out_5(fmap_out_5),   .fmap_out_6(fmap_out_6),   .fmap_out_7(fmap_out_7),
        .fmap_out_8(fmap_out_8),   .fmap_out_9(fmap_out_9),   .fmap_out_10(fmap_out_10), .fmap_out_11(fmap_out_11),
        .fmap_out_12(fmap_out_12), .fmap_out_13(fmap_out_13), .fmap_out_14(fmap_out_14), .fmap_out_15(fmap_out_15),
        
        .valid_out(valid_out)
    );

    // --- 4. CLOCK GENERATION ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // --- 5. STIMULUS PROCESS (DRIVER) ---
    initial begin
        // Mở file input (Output của lớp S2)
        file_in = $fopen("output_s2_log.txt", "r");
        if (file_in == 0) begin
            $display("ERROR: Không tìm thấy file input 'output_s2_log.txt'");
            $stop;
        end

        // Mở file output để ghi kết quả C3
        file_out = $fopen("output_c3_log.txt", "w");

        // Reset hệ thống
        rst_n = 0;
        valid_in = 0;
        fmap_in_0 = 0; fmap_in_1 = 0; fmap_in_2 = 0;
        fmap_in_3 = 0; fmap_in_4 = 0; fmap_in_5 = 0;
        
        #100; // Giữ reset trong 100ns
        rst_n = 1;
        #20;

        $display("--- BẮT ĐẦU MÔ PHỎNG LAYER C3 ---");

        // Vòng lặp đọc file
        while (!$feof(file_in)) begin
            @(negedge clk); // Đưa dữ liệu vào tại cạnh xuống để setup time an toàn
            
            // Đọc 6 giá trị từ 1 dòng của file S2
            scan_res = $fscanf(file_in, "%d %d %d %d %d %d\n", 
                               fmap_in_0, fmap_in_1, fmap_in_2, 
                               fmap_in_3, fmap_in_4, fmap_in_5);
            
            if (scan_res == 6) begin
                valid_in = 1;
            end else begin
                valid_in = 0; // Nếu dòng lỗi hoặc hết file
            end
        end

        // Kết thúc dữ liệu vào
        @(negedge clk);
        valid_in = 0;
        
        // Chờ Pipeline xả hết dữ liệu ra
        // C3 có độ trễ lớn (Line Buffer 5 dòng + Pipeline tính toán ~10 nhịp)
        // Nên chờ khoảng 1000 clock cho chắc chắn
        #10000; 

        $display("--- MÔ PHỎNG HOÀN TẤT ---");
        $display("Kết quả đã được ghi vào 'output_c3_log.txt'");
        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

    // --- 6. MONITOR PROCESS (GHI FILE) ---
    always @(posedge clk) begin
        if (valid_out) begin
            // Ghi 16 giá trị output trên 1 dòng
            $fwrite(file_out, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", 
                fmap_out_0, fmap_out_1, fmap_out_2, fmap_out_3,
                fmap_out_4, fmap_out_5, fmap_out_6, fmap_out_7,
                fmap_out_8, fmap_out_9, fmap_out_10, fmap_out_11,
                fmap_out_12, fmap_out_13, fmap_out_14, fmap_out_15
            );
        end
    end

endmodule