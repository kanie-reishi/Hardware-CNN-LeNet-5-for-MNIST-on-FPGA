`timescale 1ns/1ps

module tb_pooling_layer_s4;

    // 1. PARAMETERS
    parameter WIDTH = 32;
    parameter IMG_WIDTH = 10;

    // 2. SIGNALS
    reg clk;
    reg rst_n;
    reg valid_in;
    
    // 16 Inputs
    reg signed [WIDTH-1:0] in_0,  in_1,  in_2,  in_3;
    reg signed [WIDTH-1:0] in_4,  in_5,  in_6,  in_7;
    reg signed [WIDTH-1:0] in_8,  in_9,  in_10, in_11;
    reg signed [WIDTH-1:0] in_12, in_13, in_14, in_15;

    // 16 Outputs
    wire signed [WIDTH-1:0] out_0,  out_1,  out_2,  out_3;
    wire signed [WIDTH-1:0] out_4,  out_5,  out_6,  out_7;
    wire signed [WIDTH-1:0] out_8,  out_9,  out_10, out_11;
    wire signed [WIDTH-1:0] out_12, out_13, out_14, out_15;
    
    wire valid_out;

    // File handling
    integer file_in, file_out;
    integer scan_res;

    // 3. INSTANTIATE DUT
    pooling_layer_s4 #(
        .WIDTH(WIDTH),
        .IMG_WIDTH(IMG_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        // Inputs
        .feat_map_in_0(in_0),   .feat_map_in_1(in_1),   .feat_map_in_2(in_2),   .feat_map_in_3(in_3),
        .feat_map_in_4(in_4),   .feat_map_in_5(in_5),   .feat_map_in_6(in_6),   .feat_map_in_7(in_7),
        .feat_map_in_8(in_8),   .feat_map_in_9(in_9),   .feat_map_in_10(in_10), .feat_map_in_11(in_11),
        .feat_map_in_12(in_12), .feat_map_in_13(in_13), .feat_map_in_14(in_14), .feat_map_in_15(in_15),
        .valid_in(valid_in),
        // Outputs
        .pooled_out_0(out_0),   .pooled_out_1(out_1),   .pooled_out_2(out_2),   .pooled_out_3(out_3),
        .pooled_out_4(out_4),   .pooled_out_5(out_5),   .pooled_out_6(out_6),   .pooled_out_7(out_7),
        .pooled_out_8(out_8),   .pooled_out_9(out_9),   .pooled_out_10(out_10), .pooled_out_11(out_11),
        .pooled_out_12(out_12), .pooled_out_13(out_13), .pooled_out_14(out_14), .pooled_out_15(out_15),
        .valid_out(valid_out)
    );

    // 4. CLOCK
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    // 5. STIMULUS
    initial begin
        file_in = $fopen("output_c3_log.txt", "r");
        if (file_in == 0) begin
            $display("ERROR: Cannot open output_c3_log.txt");
            $finish;
        end
        file_out = $fopen("output_s4_log.txt", "w");

        // Init
        rst_n = 0; valid_in = 0;
        in_0 = 0; in_1 = 0; // ... (Giản lược, gán hết bằng 0)
        #100;
        rst_n = 1;
        #20;

        $display("START S4 SIMULATION...");

        while (!$feof(file_in)) begin
            @(negedge clk);
            scan_res = $fscanf(file_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7,
                in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15);
            
            if (scan_res == 16) valid_in = 1;
            else valid_in = 0;
        end

        @(negedge clk);
        valid_in = 0;
        
        // Chờ xử lý xong (ảnh 10x10 rất nhanh)
        #500;
        $display("DONE. Check output_s4_log.txt");
        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

    // 6. MONITOR
    always @(posedge clk) begin
        if (valid_out) begin
            $fwrite(file_out, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7,
                out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15);
        end
    end

endmodule