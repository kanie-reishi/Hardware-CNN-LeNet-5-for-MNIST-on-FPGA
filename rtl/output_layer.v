module output_layer #(
    parameter IN_WIDTH = 32,
    parameter W_WIDTH = 8,
    parameter OUT_WIDTH = 32,
    parameter NUM_IN = 84,  // Input từ F6
    parameter NUM_OUT = 10  // 10 chữ số (0-9)
)(
    input clk,
    input rst_n,
    input valid_in,
    input wire [IN_WIDTH*NUM_IN-1:0] in_data,
    output wire [OUT_WIDTH*NUM_OUT-1:0] out_data,
    output reg valid_out
);

    // --- 1. MEMORY ---
    reg signed [IN_WIDTH-1:0] input_buffer [0:NUM_IN-1];
    
    // Weight & Bias
    (* ram_style = "distributed" *)
    reg signed [W_WIDTH-1:0] weights [0:NUM_OUT-1][0:NUM_IN-1];
    reg signed [OUT_WIDTH-1:0] biases [0:NUM_OUT-1];

    integer b;
    initial begin
        // File hex chứa 10 dòng, mỗi dòng 84 bytes
        $readmemh("out_weight.hex", weights);
        
        // Init bias (hoặc load file)
        for (b = 0; b < NUM_OUT; b = b + 1) biases[b] = 0;
        // $readmemh("out_biases.hex", biases);
    end

    // --- 2. FSM ---
    localparam IDLE = 0, COMPUTE = 1, FINISH = 2;
    reg [1:0] state;
    reg [7:0] idx; // Đếm từ 0 đến 83
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            idx <= 0;
            valid_out <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 0;
                    idx <= 0;
                    if(valid_in) begin
                        // Load toàn bộ 84 input
                        for (i = 0; i < NUM_IN; i = i + 1) begin
                            input_buffer[i] <= in_data[i*IN_WIDTH +: IN_WIDTH];
                        end
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if(idx == NUM_IN - 1) begin
                        state <= FINISH;
                    end else begin
                        idx <= idx + 1;
                    end
                end

                FINISH: begin
                    valid_out <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // --- 3. PE LOGIC (10 PEs song song) ---
    reg signed [OUT_WIDTH-1:0] pe_results [0:NUM_OUT-1];
    
    // Saturation Constants
    localparam signed [OUT_WIDTH-1:0] MAX_POS = 2147483647;
    localparam signed [OUT_WIDTH-1:0] MIN_NEG = -2147483648;

    genvar pe_id;
    generate
        for (pe_id = 0; pe_id < NUM_OUT; pe_id = pe_id + 1) begin : PE_GEN
            
            reg signed [63:0] acc;
            wire signed [W_WIDTH-1:0] w_val;
            wire signed [IN_WIDTH-1:0] in_val;
            
            assign w_val = weights[pe_id][idx];
            assign in_val = input_buffer[idx];

            always @(posedge clk) begin
                if(state == IDLE && valid_in) begin
                    acc <= biases[pe_id];
                end
                else if(state == COMPUTE) begin
                    acc <= acc + (w_val * in_val);
                end
                else if(state == FINISH) begin
                    // [QUAN TRỌNG] Tầng cuối: KHÔNG CÓ RELU
                    // Chỉ Saturation để giữ trong 32-bit
                    
                    if (acc > MAX_POS) pe_results[pe_id] <= MAX_POS;
                    else if (acc < MIN_NEG) pe_results[pe_id] <= MIN_NEG;
                    else pe_results[pe_id] <= acc[OUT_WIDTH-1:0];
                end
            end
        end
    endgenerate

    // --- 4. PACK OUTPUT ---
    genvar out_idx;
    generate
        for (out_idx = 0; out_idx < NUM_OUT; out_idx = out_idx + 1) begin : PACK_OUT
            assign out_data[out_idx*OUT_WIDTH +: OUT_WIDTH] = pe_results[out_idx];
        end
    endgenerate

endmodule