`timescale 1ns/1ps

module tb_conv_layer_c5;

    // PARAMETERS
    parameter IN_WIDTH = 32;
    parameter OUT_WIDTH = 32;
    parameter NUM_PE = 60;

    // SIGNALS
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [IN_WIDTH*16-1:0] flat_in_data;

    wire [OUT_WIDTH*120-1:0] flat_out_data;
    wire valid_out;

    // FILE HANDLES
    integer file_in, file_out;
    integer scan_res;
    
    // VARIABLES
    reg signed [IN_WIDTH-1:0] in_ch [0:15];
    reg signed [OUT_WIDTH-1:0] out_ch [0:119];
    integer i, k;

    // INSTANTIATE DUT
    conv_layer_c5 #(
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .NUM_PE(NUM_PE)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .flat_in_data(flat_in_data),
        .in_valid(valid_in),
        .flat_out_data(flat_out_data),
        .valid_out(valid_out)
    );

    // CLOCK GEN (100 MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // STIMULUS (INPUT DRIVER)
    initial begin
        file_in = $fopen("output_s4_log.txt", "r");
        if (file_in == 0) begin
            $display("ERROR: Cannot open input file.");
            $finish;
        end
        file_out = $fopen("output_c5_log.txt", "w");
        
        // Init
        rst_n = 0;
        valid_in = 0;
        flat_in_data = 0;
        
        #100;
        rst_n = 1;
        #20;

        $display("--- START SIMULATION ---");

        // Input Loop: Lái dữ liệu tại cạnh xuống để tránh Race Condition với Clock của DUT
        while (!$feof(file_in)) begin
            // Đọc dữ liệu từ file trước
            scan_res = $fscanf(file_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                in_ch[0], in_ch[1], in_ch[2], in_ch[3],
                in_ch[4], in_ch[5], in_ch[6], in_ch[7],
                in_ch[8], in_ch[9], in_ch[10], in_ch[11],
                in_ch[12], in_ch[13], in_ch[14], in_ch[15]);

            // Đợi cạnh xuống để lái tín hiệu vào
            @(negedge clk); 

            if (scan_res == 16) begin
                valid_in = 1;
                for (k = 0; k < 16; k = k + 1) begin
                    flat_in_data[k*32 +: 32] = in_ch[k];
                end
            end else begin
                valid_in = 0;
            end
        end

        // Kết thúc input
        @(negedge clk);
        valid_in = 0;
        
        // Chờ đủ lâu để C5 xử lý hết (Tính cả delay Pipeline và FSM)
        #5000; 
        
        $display("--- SIMULATION FINISHED ---");
        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

    // MONITOR (OUTPUT SAMPLER) - CRITICAL FIX
    // Sử dụng @(negedge clk) để đảm bảo dữ liệu từ DUT đã ổn định hoàn toàn
    always @(negedge clk) begin
        if (valid_out) begin
            $display("Time %t: Capture valid output!", $time);
            
            // 1. Unpack dữ liệu từ Bus ngay lập tức
            for (i = 0; i < 120; i = i + 1) begin
                out_ch[i] = flat_out_data[i*32 +: 32];
            end
            
            // 2. Ghi vào file ngay trong block này
            for (i = 0; i < 119; i = i + 1) begin
                $fwrite(file_out, "%d ", out_ch[i]);
            end
            $fwrite(file_out, "%d\n", out_ch[119]); 
        end
    end

endmodule