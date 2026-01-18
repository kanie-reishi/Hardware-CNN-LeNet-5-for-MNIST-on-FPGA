module conv_pe_c5 #(
    parameter IN_WIDTH = 32,
    parameter W_WIDTH = 8,
    parameter OUT_WIDTH = 32
)(
    input clk,
    input rst_n,
    input en_accum,       // Cho phép cộng dồn
    input clr_accum,      // Reset bộ cộng dồn (khi bắt đầu ảnh mới)
    input last_in,        // Tín hiệu chỉ ra pixel cuối cùng trong cửa sổ 5x5 (để biết khi nào kết thúc cộng dồn)
    // INPUT: 16 pixel từ S4 (tại 1 thời điểm)
    input [IN_WIDTH*16-1:0] pixels_flat, 
    
    // WEIGHTS: 16 trọng số tương ứng với vị trí pixel hiện tại
    input [W_WIDTH*16-1:0] weights_flat,
    
    input signed [OUT_WIDTH-1:0] bias,
    
    output reg signed [OUT_WIDTH-1:0] conv_out,
    output wire valid_out
);

    // 1. UNPACK INPUTS & WEIGHTS
    wire signed [IN_WIDTH-1:0] pixels [0:15];
    wire signed [W_WIDTH-1:0]  ws [0:15];
    
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : UNPACK
            assign pixels[i] = pixels_flat[i*IN_WIDTH +: IN_WIDTH];
            assign ws[i]     = weights_flat[i*W_WIDTH +: W_WIDTH];
        end
    endgenerate

    // 2. PARALLEL MULTIPLY (16 bộ nhân)
    reg signed [IN_WIDTH+W_WIDTH+10-1:0] mult_res [0:15];
    integer k;
    always @(posedge clk) begin
        for (k = 0; k < 16; k = k + 1) begin
            mult_res[k] <= pixels[k] * ws[k];
        end
    end

    // 3. PIPELINED ADDER TREE (Tối ưu timing: 16 -> 8 -> 4 -> 2 -> 1)
    // Latency tăng thêm 3 chu kỳ so với bản cũ (Total 4 cycles from mult_res to sum_step)
    reg signed [IN_WIDTH+W_WIDTH+10+1-1:0] sum_st1 [0:7]; // Stage 1
    reg signed [IN_WIDTH+W_WIDTH+10+2-1:0] sum_st2 [0:3]; // Stage 2
    reg signed [IN_WIDTH+W_WIDTH+10+3-1:0] sum_st3 [0:1]; // Stage 3
    reg signed [IN_WIDTH+W_WIDTH+10+4-1:0] sum_step;      // Stage 4 (Final)
    
    always @(posedge clk) begin
        // Stage 1: 16 -> 8
        sum_st1[0] <= mult_res[0]  + mult_res[1];  sum_st1[1] <= mult_res[2]  + mult_res[3];
        sum_st1[2] <= mult_res[4]  + mult_res[5];  sum_st1[3] <= mult_res[6]  + mult_res[7];
        sum_st1[4] <= mult_res[8]  + mult_res[9];  sum_st1[5] <= mult_res[10] + mult_res[11];
        sum_st1[6] <= mult_res[12] + mult_res[13]; sum_st1[7] <= mult_res[14] + mult_res[15];

        // Stage 2: 8 -> 4
        sum_st2[0] <= sum_st1[0] + sum_st1[1];     sum_st2[1] <= sum_st1[2] + sum_st1[3];
        sum_st2[2] <= sum_st1[4] + sum_st1[5];     sum_st2[3] <= sum_st1[6] + sum_st1[7];

        // Stage 3: 4 -> 2
        sum_st3[0] <= sum_st2[0] + sum_st2[1];     sum_st3[1] <= sum_st2[2] + sum_st2[3];

        // Stage 4: 2 -> 1
        sum_step   <= sum_st3[0] + sum_st3[1];
    end

    // Delay Control Signals để khớp với Pipeline (Delay 5 chu kỳ)
    reg [4:0] en_accum_d, clr_accum_d;
    reg [5:0] last_in_d;
    
    always @(posedge clk) begin
        // Shift 5 lần
        en_accum_d  <= {en_accum_d[4:0], en_accum};
        clr_accum_d <= {clr_accum_d[4:0], clr_accum};
        last_in_d   <= {last_in_d[4:0], last_in};
    end

    wire en_accum_delayed  = en_accum_d[4];
    wire clr_accum_delayed = clr_accum_d[4];

    wire trigger_output = last_in_d[5]; // Kích hoạt xuất kết quả khi xử lý xong pixel cuối cùng
    assign valid_out = trigger_output;
    // 4. ACCUMULATOR (Cộng dồn theo thời gian - 25 nhịp)
    // Cần Accumulator lớn để tránh tràn số (ví dụ 48-bit)
    reg signed [57:0] accumulator;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
        end else begin
            if (clr_accum_delayed) begin
                accumulator <= sum_step; // FIX: Load giá trị đầu tiên thay vì reset về 0
            end else if (en_accum_delayed) begin
                // Cộng dồn kết quả của bước hiện tại vào tổng
                accumulator <= accumulator + sum_step;
            end
        end
    end
    // Define Max/Min for saturation (Clamping) to prevent overflow wrapping
    localparam signed [OUT_WIDTH-1:0] MAX_POS = {1'b0, {(OUT_WIDTH-1){1'b1}}}; // +2,147,483,647
    localparam signed [OUT_WIDTH-1:0] MIN_NEG = {1'b1, {(OUT_WIDTH-1){1'b0}}}; // -2,147,483,648
    wire signed [57:0] final_val = accumulator + bias;
    // 5. OUTPUT LOGIC WITH SATURATION & VALID SIGNAL
    // Output Register Logic
    always @(posedge clk) begin
        if (!rst_n) begin
            conv_out <= 0;
        end else begin

            // Capture giá trị Accumulator tại đúng thời điểm kết thúc
            if (trigger_output) begin
                // Logic Saturation
                if (final_val > MAX_POS) begin
                      conv_out <= MAX_POS;
                end else if (final_val < MIN_NEG) begin
                    conv_out <= MIN_NEG;
                end else begin
                    conv_out <= final_val[OUT_WIDTH-1:0];
                end
            end
        end
    end

endmodule