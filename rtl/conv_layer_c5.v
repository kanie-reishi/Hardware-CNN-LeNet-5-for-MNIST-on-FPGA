module conv_layer_c5 #(
    parameter IN_WIDTH = 32,
    parameter W_WIDTH = 8,
    parameter OUT_WIDTH = 32,
    parameter NUM_PE = 60 // Use 60 processing elements for C5 layer (960 DSP for ZCU102)
)(
    input clk,
    input rst_n,

    // INPUT STREAM (From S4 - 16 channels)
    // Flattened 16 channels of IN_WIDTH (16 * IN_WIDTH bits)
    input [IN_WIDTH*16-1:0] flat_in_data,
    input in_valid,

    // OUTPUT STREAM (To F6 - 120 channels)
    // Flattened 120 channels of OUT_WIDTH (120 * OUT_WIDTH bits)
    output wire [OUT_WIDTH*120-1:0] flat_out_data,
    output reg valid_out
);
    // --- 1. PING-PONG MEMORY FOR INPUT FEATURE MAPS ---
    (* ram_style = "distributed" *) 
    reg [IN_WIDTH*16-1:0] pp_buffer [0:1][0:24]; // 2 ping-pong buffers, each with 25 rows of 16 channels

    reg pp_write_sel; // 0: write to buffer 0, 1: write to buffer 1
    reg pp_read_sel;  // 0: read from buffer 0, 1: read from buffer 1

    reg [4:0] write_ptr;
    reg [4:0] read_ptr;
    
    // Flags
    reg buffer_full; // Indicates when a buffer is full
    reg process_done; // Indicates when PE process 2 pass is done

    // FSM States
    localparam IDLE = 0,
                PASS_1 = 1,
                PASS_2 = 2,
                WAIT_SWAP = 3; // Wait for ping-pong sync
    reg [1:0] state;
    // --- 2. WRITE LOGIC ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pp_write_sel <= 0;
            write_ptr <= 0;
            buffer_full <= 0;
        end else if (in_valid) begin
            pp_buffer[pp_write_sel][write_ptr] <= flat_in_data;
            if (write_ptr == 24) begin
                write_ptr <= 0;
                buffer_full <= 1;
            end else begin
                write_ptr <= write_ptr + 1;
            end
        end else if (state == PASS_1 || state == PASS_2) begin
            buffer_full <= 0; // reset sau khi FSM bắt đầu xử lý buffer này
        end
    end
    // --- 3. CONTROL LOGIC & FSM ---
    // Weight & Bias Memory
    (* ram_style = "block" *)
    reg signed [W_WIDTH-1:0] weight_mem [0:47999]; // 120 filters * 16 channels * 25 weights = 48000 weights
    reg signed [31:0] bias_mem [0:119]; // 120 biases
    integer b;
    initial begin
        $readmemh("c5_weight.hex", weight_mem);
        // $readmemh("c5_biases.mem", bias_mem);
        // Set biases to zero for simplicity
        for (b = 0; b < 120; b = b + 1) begin
            bias_mem[b] = 32'sd0;
        end
    end
    // FSM Control
    reg clr_accum, en_accum;
    reg current_group; // 0: for Pass 1, 1: for Pass 2
    reg [3:0] drain_cnt; // Đếm số nhịp chờ pipeline xả

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            read_ptr <= 0;
            pp_read_sel <= 1; // Khởi động đọc Buff 1 (lúc đầu rác, nhưng sẽ swap ngay)
            // Lưu ý: read_sel phải ngược write_sel sau khi swap
            current_group <= 0;
            valid_out <= 0;
            clr_accum <= 0;
            en_accum <= 0;
            drain_cnt <= 0; // Thêm dòng này cho chuẩn Best Practice
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 0;
                    // Chờ tín hiệu Buffer Full lần đầu tiên
                    if (buffer_full) begin
                        // SWAP NGAY LẬP TỨC
                        pp_read_sel <= pp_write_sel; // Đọc cái vừa ghi xong
                        pp_write_sel <= ~pp_write_sel; // Ghi sang cái kia
                        
                        state <= PASS_1;
                        clr_accum <= 1;
                        read_ptr <= 0;
                    end
                end

                PASS_1: begin
                    valid_out <= 0;
                    clr_accum <= 0;
                    en_accum <= 1;
                    read_ptr <= read_ptr + 1;
                    
                    if (read_ptr == 24) begin
                        state <= PASS_2;
                        read_ptr <= 0;
                        current_group <= 1;
                        clr_accum <= 1;
                    end
                end

                PASS_2: begin
                    clr_accum <= 0;
                    en_accum <= 1;
                    read_ptr <= read_ptr + 1;
                    
                    if (read_ptr == 24) begin
                        en_accum <= 0;
                        current_group <= 0;
                        state <= WAIT_SWAP; // Tính xong, chờ xem buffer kia đầy chưa
                        drain_cnt <= 0; // Reset đếm nhịp chờ
                    end
                end
                
                WAIT_SWAP: begin
                    // BƯỚC 1: CHỜ XẢ PIPELINE (6 Cycles)
                    // Cần khớp với độ trễ capture_d (bit 5) là 6 nhịp
                    if (drain_cnt < 6) begin
                        drain_cnt <= drain_cnt + 1;
                        valid_out <= 0; 
                    end
                    // BƯỚC 2: BẬT VALID (Tại nhịp thứ 6, data đã vào final_results)
                    else if (drain_cnt == 6) begin
                        valid_out <= 1;     // Bắn tín hiệu ra F6
                        drain_cnt <= drain_cnt + 1; // Tăng để không vào lại nhánh này
                    end
                    // BƯỚC 3: TẮT VALID VÀ CHỜ PING-PONG SYNC
                    else begin
                        valid_out <= 0; // Chỉ pulse 1 nhịp
                        
                        if (buffer_full) begin
                            // Swap Logic
                            pp_read_sel <= pp_write_sel;
                            pp_write_sel <= ~pp_write_sel;
                            
                            state <= PASS_1;
                            clr_accum <= 1;
                            read_ptr <= 0;
                        end
                    end
                end
            endcase
        end
    end

    // --- 4. DATA FETCHING ---
    wire [IN_WIDTH*16-1:0] current_pixels;
    assign current_pixels = pp_buffer[pp_read_sel][read_ptr];

    // --- 5. PROCESSING ELEMENTS ---

    // Temp Arrays
    wire signed [OUT_WIDTH-1:0] pe_out_array [0:NUM_PE-1];

    // Output Registers
    reg signed [OUT_WIDTH-1:0] final_results [0:119];

    genvar pe_idx;
    generate
        for(pe_idx = 0; pe_idx < NUM_PE; pe_idx = pe_idx + 1) begin : PE_ARRAY
            // Weight Fetching
            wire [W_WIDTH*16-1:0] weight_flat;
            genvar w_idx;
            for(w_idx = 0; w_idx < 16; w_idx = w_idx + 1) begin : WEIGHT_FETCH
                assign weight_flat[(w_idx*W_WIDTH) +: W_WIDTH] = weight_mem[(current_group * NUM_PE + pe_idx) * 16 * 25 + (read_ptr * 16) + w_idx];
            end
            // Bias Fetching
            wire signed [OUT_WIDTH-1:0] bias;
            assign bias = bias_mem[current_group * NUM_PE + pe_idx];

            // Instantiate PE
            wire signed [OUT_WIDTH-1:0] pe_out;
            wire pe_valid_wire;
            wire pe_last_in = (read_ptr == 24) && en_accum;

            conv_pe_c5 #(
                .IN_WIDTH(IN_WIDTH),
                .W_WIDTH(W_WIDTH),
                .OUT_WIDTH(OUT_WIDTH)
            ) conv_pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                .pixels_flat(current_pixels),
                .weights_flat(weight_flat),
                .clr_accum(clr_accum),
                .en_accum(en_accum),
                .last_in(pe_last_in),
                .bias(bias),
                .conv_out(pe_out),
                .valid_out(pe_valid_wire)
            );
            assign pe_out_array[pe_idx] = pe_out;
            // --- 6. COLLECT OUTPUTS ---
            always @(posedge clk) begin
                // Capture khi PE báo valid
                    if(pe_valid_wire) begin
                        final_results[pe_idx] <= pe_out;
                    end else begin
                        final_results[NUM_PE + pe_idx] <= pe_out;
                    end
                end
            end
    endgenerate
    // --- 7. FLATTEN OUTPUT ---
    genvar out_idx;
    generate
        for(out_idx = 0; out_idx < 120; out_idx = out_idx + 1) begin : FLATTEN
            assign flat_out_data[(out_idx*OUT_WIDTH) +: OUT_WIDTH] = final_results[out_idx];
        end
    endgenerate
endmodule 