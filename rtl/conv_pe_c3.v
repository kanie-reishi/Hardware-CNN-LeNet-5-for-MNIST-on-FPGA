// -- Multi-channel convolution processing element: 6 input window channels -> 1 output channel --
// Conv layer C3 - PE unit
// Each PE processes 6 input channels and produces 1 output channel
// 16 PE units in parallel for 16 output channels
// Each input channel has a 5x5 kernel (25 weights) + bias
module conv_pe_c3 #(
    parameter IN_WIDTH = 24, // Input feature map pixel width
    parameter W_WIDTH = 8,   // Weight width
    parameter OUT_WIDTH = 32  // Output feature map pixel width
)(
    input clk,
    input rst_n,
    // 6 input channels' 5x5 windows
    input [IN_WIDTH*25-1:0] win_flat_ch0,
    input [IN_WIDTH*25-1:0] win_flat_ch1,
    input [IN_WIDTH*25-1:0] win_flat_ch2,
    input [IN_WIDTH*25-1:0] win_flat_ch3,
    input [IN_WIDTH*25-1:0] win_flat_ch4,
    input [IN_WIDTH*25-1:0] win_flat_ch5,
    input valid_in,
    // Weights + bias for channel
    input [W_WIDTH*150-1:0] weight_flat, // 6 channels * 25 weights
    input signed [OUT_WIDTH-1:0] bias,
    // Output channel
    output reg signed [OUT_WIDTH-1:0] conv_out,
    output reg valid_out

);
    // 1. Declare Multiplication registers
    reg signed [IN_WIDTH+W_WIDTH-1:0] mult_res [0:149];

    // 2. Declare Partial Sum registers Adder Tree
    // Bit widths are increased by 1 at each 2-input adder stage to prevent overflow.
    // Total of 150 products requires log2(150) approx 8 extra bits for the final sum.
    localparam SUM_WIDTH = IN_WIDTH + W_WIDTH + 8;
    reg signed [IN_WIDTH+W_WIDTH+1-1:0] psum_stage1 [0:74]; // Stage 2: 75 partial sums
    reg signed [IN_WIDTH+W_WIDTH+2-1:0] psum_stage2 [0:37]; // Stage 3: 38 partial sums
    reg signed [IN_WIDTH+W_WIDTH+3-1:0] psum_stage3 [0:18]; // Stage 4: 19 partial sums
    reg signed [IN_WIDTH+W_WIDTH+4-1:0] psum_stage4 [0:9];  // Stage 5: 10 partial sums
    reg signed [IN_WIDTH+W_WIDTH+5-1:0] psum_stage5 [0:4];  // Stage 6: 5 partial sums
    // --- Timing Optimization: Break down final additions into more stages ---
    reg signed [IN_WIDTH+W_WIDTH+6-1:0] psum_stage6 [0:2];  // Stage 7: 3 partial sums
    reg signed [IN_WIDTH+W_WIDTH+7-1:0] psum_stage7 [0:1];  // Stage 8: 2 partial sums
    reg signed [IN_WIDTH+W_WIDTH+8-1:0] psum_stage8;        // Stage 9: Final product sum
    reg signed [SUM_WIDTH+1-1:0]        final_sum;          // Stage 10: Final sum with bias

    // 3. INTERNAL UNPACKED ARRAYS FOR WINDOWS AND WEIGHTS
    wire signed [IN_WIDTH-1:0] window_ch0 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch1 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch2 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch3 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch4 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch5 [0:24];
    wire signed [W_WIDTH-1:0] weight_array [0:149];

    // Unflatten input windows and weights
    genvar idx;
    generate
        // Unflatten windows
        for(idx = 0; idx < 25; idx = idx + 1) begin : UNPACK_WINDOWS

            assign window_ch0[idx] = win_flat_ch0[(idx*IN_WIDTH) +: IN_WIDTH];
            assign window_ch1[idx] = win_flat_ch1[(idx*IN_WIDTH) +: IN_WIDTH];
            assign window_ch2[idx] = win_flat_ch2[(idx*IN_WIDTH) +: IN_WIDTH];
            assign window_ch3[idx] = win_flat_ch3[(idx*IN_WIDTH) +: IN_WIDTH];
            assign window_ch4[idx] = win_flat_ch4[(idx*IN_WIDTH) +: IN_WIDTH];
            assign window_ch5[idx] = win_flat_ch5[(idx*IN_WIDTH) +: IN_WIDTH];
        end
        // Unflatten weights
        for(idx = 0; idx < 150; idx = idx + 1) begin : UNPACK_WEIGHTS

            assign weight_array[idx] = weight_flat[idx*W_WIDTH +: W_WIDTH];
        end
    endgenerate

    // --- PIPELINE CONTROL ---
    reg [9:0] valid_pipe; // 11-stage pipeline for improved timing
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            valid_out <= 0;
        end else begin
            valid_pipe <= {valid_pipe[8:0], valid_in}; // Shift Left
            valid_out <= valid_pipe[9];
        end
    end
    // --- STAGE 1: MULTIPLICATION ---
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < 25; i = i + 1) begin
            // Single multiplication for 6 channels
            mult_res[i]      <= $signed(window_ch0[i]) * weight_array[i];
            mult_res[i+25]   <= $signed(window_ch1[i]) * weight_array[i+25];
            mult_res[i+50]   <= $signed(window_ch2[i]) * weight_array[i+50];
            mult_res[i+75]   <= $signed(window_ch3[i]) * weight_array[i+75];
            mult_res[i+100]  <= $signed(window_ch4[i]) * weight_array[i+100];
            mult_res[i+125]  <= $signed(window_ch5[i]) * weight_array[i+125];
        end
    end
    // --- PIPELINED ADDER TREE (STAGES 2-10) ---
    always @(posedge clk) begin
        // Adder Tree level 1 (Stage 2)
        for (i = 0; i < 75; i = i + 1) begin
            psum_stage1[i] <= mult_res[2*i] + mult_res[2*i + 1];
        end
    end
    always @(posedge clk) begin
        // Adder Tree level 2 (Stage 3)
        for (i = 0; i < 37; i = i + 1) begin
            psum_stage2[i] <= psum_stage1[2*i] + psum_stage1[2*i + 1];
        end
        // Handle odd one
        psum_stage2[37] <= psum_stage1[74];
    end
    always @(posedge clk) begin
        // Adder Tree level 3 (Stage 4)
        for (i = 0; i < 19; i = i + 1) begin
            psum_stage3[i] <= psum_stage2[2*i] + psum_stage2[2*i + 1];
        end
    end
    always @(posedge clk) begin
        // Adder Tree level 4 (Stage 5): 19 inputs -> 10 outputs
        for (i = 0; i < 9; i = i + 1) begin
            psum_stage4[i] <= psum_stage3[2*i] + psum_stage3[2*i + 1];
        end
        // Handle odd one
        psum_stage4[9] <= psum_stage3[18];
    end
    always @(posedge clk) begin
        // Adder Tree level 5 (Stage 6): 10 inputs -> 5 outputs
        for (i = 0; i < 5; i = i + 1) begin
            psum_stage5[i] <= psum_stage4[2*i] + psum_stage4[2*i + 1];
        end
    end
    // --- Timing Optimized Final Adder Stages ---
    always @(posedge clk) begin
        // Adder Tree level 6 (Stage 7): 5 inputs -> 3 outputs
        psum_stage6[0] <= psum_stage5[0] + psum_stage5[1];
        psum_stage6[1] <= psum_stage5[2] + psum_stage5[3];
        psum_stage6[2] <= psum_stage5[4]; // Pass through
    end
    always @(posedge clk) begin
        // Adder Tree level 7 (Stage 8): 3 inputs -> 2 outputs
        psum_stage7[0] <= psum_stage6[0] + psum_stage6[1];
        psum_stage7[1] <= psum_stage6[2]; // Pass through
    end
    always @(posedge clk) begin
        // Adder Tree level 8 (Stage 9): 2 inputs -> 1 output (Final product sum)
        psum_stage8 <= psum_stage7[0] + psum_stage7[1];
    end
    always @(posedge clk) begin
        // Adder Tree level 9 (Stage 10): Add bias
        final_sum <= psum_stage8 + bias;
    end

    // --- STAGE 11: OUTPUT ASSIGNMENT & TRUNCATION ---
    // Define Max/Min for saturation (Clamping) to prevent overflow wrapping
    localparam signed [OUT_WIDTH-1:0] MAX_POS = {1'b0, {(OUT_WIDTH-1){1'b1}}}; // +2,147,483,647
    localparam signed [OUT_WIDTH-1:0] MIN_NEG = {1'b1, {(OUT_WIDTH-1){1'b0}}}; // -2,147,483,648

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_out <= 0;
        end else begin
            // Saturation Logic: Check if final_sum exceeds 32-bit range
            if (final_sum > MAX_POS)      conv_out <= MAX_POS;
            else if (final_sum < MIN_NEG) conv_out <= MIN_NEG;
            else                          conv_out <= final_sum[OUT_WIDTH-1:0];
        end
    end
endmodule