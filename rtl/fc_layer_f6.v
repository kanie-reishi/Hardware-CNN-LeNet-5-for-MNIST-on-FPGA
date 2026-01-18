module fc_layer_f6 #(
    parameter IN_WIDTH = 32, // Input width from C5
    parameter W_WIDTH = 8, // Weights from F6 - 8 bit
    parameter OUT_WIDTH = 32, // Output to Output layer
    parameter NUM_IN = 120, // Number of inputs from C5
    parameter NUM_OUT = 84 // Number of neurons in F6
)(
    input clk,
    input rst_n,

    // --- INPUT INTERFACE ---
    input valid_in,
    input wire [IN_WIDTH*NUM_IN-1:0] in_data, // Flattened input data from C5

    // --- OUTPUT INTERFACE ---
    output wire [OUT_WIDTH*NUM_OUT-1:0] out_data, // Flattened output data to Output layer
    output reg valid_out
);
    // -- 1. Input buffer, weights and biases memory --
    // Input buffer: stores input feature map from C5 to use in 120 clocks
    reg signed [IN_WIDTH-1:0] input_buffer [0:NUM_IN-1];

    // Weights memory: stores weights for F6 layer
    // Use distributed RAM for weights because we need to reading 84 weights in parallel
    (* ram_style = "distributed" *)
    reg signed [W_WIDTH-1:0] weights [0:NUM_OUT-1][0:NUM_IN-1];

    // Biases memory: stores biases for F6 layer
    reg signed [IN_WIDTH-1:0] biases [0:NUM_OUT-1];
    integer b;
    // Load weights and biases (this can be done via an external interface, here we just use initial block for simplicity)
    initial begin
        // Load weights and biases from files or initialize here
        $readmemh("f6_weight.hex", weights);
        // Set biases to zero for simplicity
        for (b = 0; b < NUM_OUT; b = b + 1) begin
            biases[b] = 8'sd0;
        end
        // $readmemh("f6_biases.mem", biases);
    end
    // -- 2. FSM and computation logic --
    localparam IDLE = 0,
               COMPUTE = 1,
               FINISH = 2;
    reg [1:0] state;
    reg [7:0] neuron_idx; // Index for output neurons (0 to 119)

    // Unpacking variables
    integer i;

    // --- 3. FSM Implementation ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            neuron_idx <= 0;
            valid_out <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 0;
                    neuron_idx <= 0;

                    if(valid_in) begin
                        // Load input data into input buffer
                        for (i = 0; i < NUM_IN; i = i + 1) begin
                            input_buffer[i] <= in_data[i*IN_WIDTH +: IN_WIDTH];
                        end
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // 120 loops for 120 output neurons
                    if(neuron_idx == NUM_IN - 1) begin
                        state <= FINISH;
                    end else begin
                        neuron_idx <= neuron_idx + 1;
                    end
                end

                FINISH: begin
                    // This State use to Relu and Saturation
                    valid_out <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // --- 4. PROCESSING ELEMENTS (84 PEs in parallel) ---

    reg signed [OUT_WIDTH-1:0] pe_results [0:NUM_OUT-1];
    // Define Max/Min for saturation (Clamping) to prevent overflow wrapping
    localparam signed [OUT_WIDTH-1:0] MAX_POS = {1'b0, {(OUT_WIDTH-1){1'b1}}}; // +2,147,483,647
    localparam signed [OUT_WIDTH-1:0] MIN_NEG = {1'b1, {(OUT_WIDTH-1){1'b0}}}; // -2,147,483,648
    genvar pe_idx;
    generate
        for (pe_idx = 0; pe_idx < NUM_OUT; pe_idx = pe_idx + 1) begin : PE_GEN
            // Accumulator expanded width to prevent overflow
            reg signed [63:0] acc;
            // Directly access weights and inputs from registers/ram based on current neuron and input index
            wire signed [W_WIDTH-1:0] w_val;
            wire signed [IN_WIDTH-1:0] in_val;

            // Each PE read its corresponding weight
            assign w_val = weights[pe_idx][neuron_idx];
            // All PE read the same index input value
            assign in_val = input_buffer[neuron_idx];

            // PE Computation
            always @(posedge clk) begin
                if(state == IDLE && valid_in) begin
                    // Clear accumulator at the start of computation
                    // Assign acc to bias value to save 1 plus operation
                    acc <= biases[pe_idx];
                end
                else if(state == COMPUTE) begin
                    // MAC: Multiply and Accumulate
                    acc <= acc + (w_val * in_val);
                end
                else if(state == FINISH) begin
                    // RELU & Saturation
                    // 1. ReLU: Negative values set to zero
                    if(acc < 0) begin
                        pe_results[pe_idx] <= 0;
                    end else begin
                        // 2. Saturation to fit into OUT_WIDTH
                        if (acc > MAX_POS) begin
                            pe_results[pe_idx] <= MAX_POS;
                        end else if (acc < MIN_NEG) begin
                            pe_results[pe_idx] <= MIN_NEG;
                        end else begin
                            pe_results[pe_idx] <= acc[OUT_WIDTH-1:0];
                        end
                    end
                end
            end
        end
    endgenerate

    // --- 5. OUTPUT LOGIC ---
    genvar out_idx;
    generate
        for (out_idx = 0; out_idx < NUM_OUT; out_idx = out_idx + 1) begin : OUTPUT_PACK
            assign out_data[out_idx*OUT_WIDTH +: OUT_WIDTH] = pe_results[out_idx];
        end
    endgenerate
endmodule