`timescale 1ns/1ps

module tb_fc_layer_f6;

    // --- PARAMETERS ---
    parameter IN_WIDTH = 32;
    parameter W_WIDTH = 8;
    parameter OUT_WIDTH = 32;
    parameter NUM_IN = 120;
    parameter NUM_OUT = 84;

    // --- SIGNALS ---
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [IN_WIDTH*NUM_IN-1:0] in_data; // Bus input lớn
    
    wire [OUT_WIDTH*NUM_OUT-1:0] out_data;
    wire valid_out;

    // --- VARIABLES ---
    reg signed [IN_WIDTH-1:0] in_array [0:NUM_IN-1];
    reg signed [OUT_WIDTH-1:0] out_array [0:NUM_OUT-1];
    
    // File handles
    integer f_in, f_out; 
    integer scan_res;
    integer i, j;
    reg signed [31:0] temp_val;

    // --- INSTANTIATE DUT ---
    fc_layer_f6 #(
        .IN_WIDTH(IN_WIDTH), .W_WIDTH(W_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .NUM_IN(NUM_IN), .NUM_OUT(NUM_OUT)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .in_data(in_data),
        .out_data(out_data),
        .valid_out(valid_out)
    );

    // --- CLOCK GEN ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    // --- MAIN TEST FLOW ---
    initial begin
        // 1. CHUẨN BỊ FILE IO
        
        // Mở file Input từ C5
        f_in = $fopen("output_c5_log.txt", "r");
        if (f_in == 0) begin
            $display("ERROR: Không tìm thấy file 'output_c5_log.txt'. Hãy chạy mô phỏng C5 hoặc Python trước!");
            $finish;
        end
        
        // Mở file Output log cho F6
        f_out = $fopen("output_f6_log.txt", "w");
        
        // [QUAN TRỌNG]: Tắt đoạn code tạo weight giả.
        // Hãy đảm bảo bạn đã có file "f6_weights.hex" thật trong thư mục project!
        /*
        f_w = $fopen("f6_weights.hex", "w");
        ... (Code cũ) ...
        */

        // 2. INIT
        rst_n = 0;
        valid_in = 0;
        in_data = 0;
        #100;
        rst_n = 1;
        #20;

        $display("--- START F6 SIMULATION WITH REAL DATA ---");

        // 3. ĐỌC FILE INPUT & PACKING
        $display("Reading data from output_c5_log.txt...");
        
        for (i = 0; i < NUM_IN; i = i + 1) begin
            // Đọc số nguyên từ file (tự động bỏ qua dấu cách/xuống dòng)
            scan_res = $fscanf(f_in, "%d", temp_val);
            
            if (scan_res == 1) begin
                in_array[i] = temp_val;
            end else begin
                $display("WARNING: File input kết thúc sớm hoặc lỗi định dạng tại index %d", i);
                in_array[i] = 0;
            end
        end
        
        // Đóng gói từ Array 120 phần tử vào Bus phẳng
        for (i = 0; i < NUM_IN; i = i + 1) begin
            in_data[i*IN_WIDTH +: IN_WIDTH] = in_array[i];
        end
        
        // Hiển thị vài giá trị đầu để debug xem đọc đúng không
        $display("Sample Input [0]: %d", in_array[0]);
        $display("Sample Input [1]: %d", in_array[1]);

        // 4. GỬI DỮ LIỆU VÀO DUT
        @(negedge clk);
        valid_in = 1; // Pulse 1 nhịp
        @(negedge clk);
        valid_in = 0;

        // 5. CHỜ KẾT QUẢ
        $display("Waiting for computation...");
        wait(valid_out);
        @(negedge clk);

        $display("--- VALID OUTPUT DETECTED ---");
        
        // Unpack Output
        for (i = 0; i < NUM_OUT; i = i + 1) begin
            out_array[i] = out_data[i*OUT_WIDTH +: OUT_WIDTH];
        end

        // 6. GHI KẾT QUẢ RA FILE
        for (i = 0; i < NUM_OUT; i = i + 1) begin
            $fwrite(f_out, "%0d ", out_array[i]);
        end
        $fwrite(f_out, "\n");

        $display("✅ Đã ghi kết quả F6 vào 'output_f6_log.txt'");
        $display("Sample Output F6 [0]: %d", out_array[0]);
        $display("Sample Output F6 [1]: %d", out_array[1]);

        // Cleanup
        $fclose(f_in);
        $fclose(f_out);
        #100;
        $finish;
    end

endmodule