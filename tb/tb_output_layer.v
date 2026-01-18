`timescale 1ns/1ps

module tb_output_layer;

    // --- PARAMETERS ---
    parameter IN_WIDTH = 32;
    parameter W_WIDTH = 8;
    parameter OUT_WIDTH = 32;
    parameter NUM_IN = 84;  // Input từ F6
    parameter NUM_OUT = 10; // 10 lớp (0-9)

    // --- SIGNALS ---
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [IN_WIDTH*NUM_IN-1:0] in_data;
    
    wire [OUT_WIDTH*NUM_OUT-1:0] out_data;
    wire valid_out;

    // --- VARIABLES ---
    reg signed [IN_WIDTH-1:0] in_array [0:NUM_IN-1];
    reg signed [OUT_WIDTH-1:0] out_array [0:NUM_OUT-1];
    
    integer f_in, f_out;
    integer scan_res;
    integer i;
    reg signed [31:0] temp_val;
    integer max_val, max_idx;
    // --- INSTANTIATE DUT ---
    output_layer #(
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

    // --- MAIN TEST ---
    initial begin
        // 1. SETUP FILES
        f_in = $fopen("output_f6_log.txt", "r");
        if (f_in == 0) begin
            $display("ERROR: Không tìm thấy 'output_f6_log.txt'. Hãy chạy mô phỏng F6 trước!");
            $finish;
        end
        
        f_out = $fopen("final_result_log.txt", "w");

        // 2. INIT
        rst_n = 0;
        valid_in = 0;
        in_data = 0;
        #100;
        rst_n = 1;
        #20;

        $display("--- START OUTPUT LAYER SIMULATION ---");

        // 3. READ INPUT (84 giá trị từ F6)
        for (i = 0; i < NUM_IN; i = i + 1) begin
            scan_res = $fscanf(f_in, "%d", temp_val);
            if (scan_res == 1) in_array[i] = temp_val;
            else in_array[i] = 0;
        end
        
        // Pack into bus
        for (i = 0; i < NUM_IN; i = i + 1) begin
            in_data[i*IN_WIDTH +: IN_WIDTH] = in_array[i];
        end

        // 4. DRIVE INPUT
        @(negedge clk);
        valid_in = 1;
        @(negedge clk);
        valid_in = 0;

        // 5. WAIT FOR RESULT
        wait(valid_out);
        @(negedge clk);

        $display("--- VALID OUTPUT DETECTED ---");
        
        // Unpack Output
        for (i = 0; i < NUM_OUT; i = i + 1) begin
            out_array[i] = out_data[i*OUT_WIDTH +: OUT_WIDTH];
        end

        // 6. PRINT & SAVE RESULTS
        $display("Final Logits (Scores):");
        for (i = 0; i < NUM_OUT; i = i + 1) begin
            $display("Class %0d: %d", i, out_array[i]);
            $fwrite(f_out, "%d ", out_array[i]);
        end
        $fwrite(f_out, "\n");
        
        // --- LOGIC TÌM ARGMAX ĐƠN GIẢN TRONG TESTBENCH ---
        // (Đây là logic phần mềm mô phỏng việc CPU sẽ làm)
        max_val = out_array[0];
        max_idx = 0;
            
        for (i = 1; i < NUM_OUT; i = i + 1) begin
            if (out_array[i] > max_val) begin
                max_val = out_array[i];
                max_idx = i;
            end
        end
        $display("\n🏆 PREDICTED DIGIT: %0d (Score: %0d)", max_idx, max_val);

        $fclose(f_in);
        $fclose(f_out);
        #100;
        $finish;
    end

endmodule