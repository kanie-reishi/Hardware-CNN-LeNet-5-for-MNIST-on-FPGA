module conv_layer_c3 #(
    parameter IN_WIDTH = 24,
    parameter W_WIDTH = 8,
    parameter OUT_WIDTH = 32 // Output feature map pixel width after conv
    //(24 bit * 8bit = 32 bit, sum accumulator 40 bit, cut back to 32 bit)
)(
    input clk,
    input rst_n,

    // Input feature map pixel stream
    // 6 input channels (from previous layer)
    input signed [IN_WIDTH-1:0] fmap_in_0,
    input signed [IN_WIDTH-1:0] fmap_in_1,
    input signed [IN_WIDTH-1:0] fmap_in_2,
    input signed [IN_WIDTH-1:0] fmap_in_3,
    input signed [IN_WIDTH-1:0] fmap_in_4,
    input signed [IN_WIDTH-1:0] fmap_in_5,
    input valid_in,

    // Output feature map pixel stream
    // 16 output channels
    output reg signed [OUT_WIDTH-1:0] fmap_out_0,
    output reg signed [OUT_WIDTH-1:0] fmap_out_1,
    output reg signed [OUT_WIDTH-1:0] fmap_out_2,
    output reg signed [OUT_WIDTH-1:0] fmap_out_3,
    output reg signed [OUT_WIDTH-1:0] fmap_out_4,
    output reg signed [OUT_WIDTH-1:0] fmap_out_5,
    output reg signed [OUT_WIDTH-1:0] fmap_out_6,
    output reg signed [OUT_WIDTH-1:0] fmap_out_7,
    output reg signed [OUT_WIDTH-1:0] fmap_out_8,
    output reg signed [OUT_WIDTH-1:0] fmap_out_9,
    output reg signed [OUT_WIDTH-1:0] fmap_out_10,
    output reg signed [OUT_WIDTH-1:0] fmap_out_11,
    output reg signed [OUT_WIDTH-1:0] fmap_out_12,
    output reg signed [OUT_WIDTH-1:0] fmap_out_13,
    output reg signed [OUT_WIDTH-1:0] fmap_out_14,
    output reg signed [OUT_WIDTH-1:0] fmap_out_15,

    output wire valid_out
);
    // -- Create 6 line buffers and window arrays for 6 input channels --
    // Each line buffer + window array handles 1 input channel
    // Instantiate 6 parallel 5x5 window array wire

    wire signed [IN_WIDTH-1:0] window_ch0 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch1 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch2 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch3 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch4 [0:24];
    wire signed [IN_WIDTH-1:0] window_ch5 [0:24];

    wire window_valid;

    // Instantiate line buffer for each input channel
    line_buffer_window_5x5 #(
        // 14 x 14 input feature map, after pooling layer S2
        // Each pixel is 24-bit
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch0 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_0[IN_WIDTH-1:0]),
        // Output 5x5 window pixels flattened
        .valid_in(valid_in),
        .p00(window_ch0[0]), .p01(window_ch0[1]), .p02(window_ch0[2]), .p03(window_ch0[3]), .p04(window_ch0[4]),
        .p10(window_ch0[5]), .p11(window_ch0[6]), .p12(window_ch0[7]), .p13(window_ch0[8]), .p14(window_ch0[9]),
        .p20(window_ch0[10]), .p21(window_ch0[11]), .p22(window_ch0[12]), .p23(window_ch0[13]), .p24(window_ch0[14]),
        .p30(window_ch0[15]), .p31(window_ch0[16]), .p32(window_ch0[17]), .p33(window_ch0[18]), .p34(window_ch0[19]),
        .p40(window_ch0[20]), .p41(window_ch0[21]), .p42(window_ch0[22]), .p43(window_ch0[23]), .p44(window_ch0[24]),
        .valid_out(window_valid)
    );
    // ... Repeat for channels 1 to 5 ...
    line_buffer_window_5x5 #(
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch1 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_1[IN_WIDTH-1:0]),
        .valid_in(valid_in),
        .p00(window_ch1[0]), .p01(window_ch1[1]), .p02(window_ch1[2]), .p03(window_ch1[3]), .p04(window_ch1[4]),
        .p10(window_ch1[5]), .p11(window_ch1[6]), .p12(window_ch1[7]), .p13(window_ch1[8]), .p14(window_ch1[9]),
        .p20(window_ch1[10]), .p21(window_ch1[11]), .p22(window_ch1[12]), .p23(window_ch1[13]), .p24(window_ch1[14]),
        .p30(window_ch1[15]), .p31(window_ch1[16]), .p32(window_ch1[17]), .p33(window_ch1[18]), .p34(window_ch1[19]),
        .p40(window_ch1[20]), .p41(window_ch1[21]), .p42(window_ch1[22]), .p43(window_ch1[23]), .p44(window_ch1[24]),
        .valid_out() // Ignore valid_out bc parallel
    );
    line_buffer_window_5x5 #(
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch2 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_2[IN_WIDTH-1:0]),
        .valid_in(valid_in),
        .p00(window_ch2[0]), .p01(window_ch2[1]), .p02(window_ch2[2]), .p03(window_ch2[3]), .p04(window_ch2[4]),
        .p10(window_ch2[5]), .p11(window_ch2[6]), .p12(window_ch2[7]), .p13(window_ch2[8]), .p14(window_ch2[9]),
        .p20(window_ch2[10]), .p21(window_ch2[11]), .p22(window_ch2[12]), .p23(window_ch2[13]), .p24(window_ch2[14]),
        .p30(window_ch2[15]), .p31(window_ch2[16]), .p32(window_ch2[17]), .p33(window_ch2[18]), .p34(window_ch2[19]),
        .p40(window_ch2[20]), .p41(window_ch2[21]), .p42(window_ch2[22]), .p43(window_ch2[23]), .p44(window_ch2[24]),
        .valid_out() // Ignore valid_out bc parallel
    );
    line_buffer_window_5x5 #(
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch3 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_3[IN_WIDTH-1:0]),
        .valid_in(valid_in),
        .p00(window_ch3[0]), .p01(window_ch3[1]), .p02(window_ch3[2]), .p03(window_ch3[3]), .p04(window_ch3[4]),
        .p10(window_ch3[5]), .p11(window_ch3[6]), .p12(window_ch3[7]), .p13(window_ch3[8]), .p14(window_ch3[9]),
        .p20(window_ch3[10]), .p21(window_ch3[11]), .p22(window_ch3[12]), .p23(window_ch3[13]), .p24(window_ch3[14]),
        .p30(window_ch3[15]), .p31(window_ch3[16]), .p32(window_ch3[17]), .p33(window_ch3[18]), .p34(window_ch3[19]),
        .p40(window_ch3[20]), .p41(window_ch3[21]), .p42(window_ch3[22]), .p43(window_ch3[23]), .p44(window_ch3[24]),
        .valid_out() // Ignore valid_out bc parallel
    );
    line_buffer_window_5x5 #(
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch4 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_4[IN_WIDTH-1:0]),
        .valid_in(valid_in),
        .p00(window_ch4[0]), .p01(window_ch4[1]), .p02(window_ch4[2]), .p03(window_ch4[3]), .p04(window_ch4[4]),
        .p10(window_ch4[5]), .p11(window_ch4[6]), .p12(window_ch4[7]), .p13(window_ch4[8]), .p14(window_ch4[9]),
        .p20(window_ch4[10]), .p21(window_ch4[11]), .p22(window_ch4[12]), .p23(window_ch4[13]), .p24(window_ch4[14]),
        .p30(window_ch4[15]), .p31(window_ch4[16]), .p32(window_ch4[17]), .p33(window_ch4[18]), .p34(window_ch4[19]),
        .p40(window_ch4[20]), .p41(window_ch4[21]), .p42(window_ch4[22]), .p43(window_ch4[23]), .p44(window_ch4[24]),
        .valid_out() // Ignore valid_out bc parallel
    );
    line_buffer_window_5x5 #(
        .IMG_WIDTH(14),
        .DATA_WIDTH(IN_WIDTH)
    ) lb_win_5x5_ch5 (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(fmap_in_5[IN_WIDTH-1:0]),
        .valid_in(valid_in),
        .p00(window_ch5[0]), .p01(window_ch5[1]), .p02(window_ch5[2]), .p03(window_ch5[3]), .p04(window_ch5[4]),
        .p10(window_ch5[5]), .p11(window_ch5[6]), .p12(window_ch5[7]), .p13(window_ch5[8]), .p14(window_ch5[9]),
        .p20(window_ch5[10]), .p21(window_ch5[11]), .p22(window_ch5[12]), .p23(window_ch5[13]), .p24(window_ch5[14]),
        .p30(window_ch5[15]), .p31(window_ch5[16]), .p32(window_ch5[17]), .p33(window_ch5[18]), .p34(window_ch5[19]),
        .p40(window_ch5[20]), .p41(window_ch5[21]), .p42(window_ch5[22]), .p43(window_ch5[23]), .p44(window_ch5[24]),
        .valid_out() // Ignore valid_out bc parallel
    );
    // -- Weights Storage for 16 output channels --
    // Each output channel has its own 5x5x6 kernel weights + bias
    // Total 16 kernels, each with 25 weights per input channel (6 channels)
    // Store weights in ROM using $readmemh (from hex files)
    reg signed [7:0] KERNEL [0:2399]; // 16 * 6 * 25 = 2400 weights
    reg signed [OUT_WIDTH-1:0] BIAS [0:15];

    initial begin
        // Load Hex files for weights and biases
        // Or hardcode them here
        $readmemh("c3_weight.hex", KERNEL);
        // Example: $readmemh("c3_bias.hex", BIAS);
        // Make biases zero for testing
        BIAS[0] = 32'sd0; BIAS[1] = 32'sd0; BIAS[2] = 32'sd0; BIAS[3] = 32'sd0;
        BIAS[4] = 32'sd0; BIAS[5] = 32'sd0; BIAS[6] = 32'sd0; BIAS[7] = 32'sd0;
        BIAS[8] = 32'sd0; BIAS[9] = 32'sd0; BIAS[10] = 32'sd0; BIAS[11] = 32'sd0;
        BIAS[12] = 32'sd0; BIAS[13] = 32'sd0; BIAS[14] = 32'sd0; BIAS[15] = 32'sd0;
    end
    // --- KHAI BÁO DÂY KẾT NỐI TRUNG GIAN ---
    // Mảng này hứng kết quả từ 16 PE trước khi đưa ra Output Ports
    wire signed [OUT_WIDTH-1:0] pe_results [0:15];
    wire [15:0] pe_valid_flags;

    // --- FLATTEN WINDOWS (Shared for all PEs) ---
    // Tối ưu: Thực hiện flatten 1 lần duy nhất ở đây thay vì lặp lại trong mỗi PE
    wire [IN_WIDTH*25-1:0] win_flat_0, win_flat_1, win_flat_2, win_flat_3, win_flat_4, win_flat_5;
    genvar p;
    generate
        for (p = 0; p < 25; p = p + 1) begin : FLATTEN_WIN_SHARED
            assign win_flat_0[(p*IN_WIDTH) +: IN_WIDTH] = window_ch0[p];
            assign win_flat_1[(p*IN_WIDTH) +: IN_WIDTH] = window_ch1[p];
            assign win_flat_2[(p*IN_WIDTH) +: IN_WIDTH] = window_ch2[p];
            assign win_flat_3[(p*IN_WIDTH) +: IN_WIDTH] = window_ch3[p];
            assign win_flat_4[(p*IN_WIDTH) +: IN_WIDTH] = window_ch4[p];
            assign win_flat_5[(p*IN_WIDTH) +: IN_WIDTH] = window_ch5[p];
        end
    endgenerate

    // --- SINH 16 PE SONG SONG (GENERATE BLOCK) ---
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : GEN_PE_ARRAY
            
            // A. Flatten Weight: Cắt 150 trọng số từ KERNEL ra và ghép lại thành 1 dây
            wire [W_WIDTH*150-1:0] current_weights_flat;
            genvar w;
            for (w = 0; w < 150; w = w + 1) begin : FLATTEN_W
                // Ghép từng weight 8-bit vào vị trí tương ứng trong vector lớn
                assign current_weights_flat[(w*W_WIDTH) +: W_WIDTH] = KERNEL[(i * 150) + w];
            end

            // C. Instantiate PE với các dây đã flatten
            conv_pe_c3 #(
                .IN_WIDTH(IN_WIDTH),
                .W_WIDTH(W_WIDTH),
                .OUT_WIDTH(OUT_WIDTH)
            ) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                
                // Nối các dây phẳng vào
                .win_flat_ch0(win_flat_0),
                .win_flat_ch1(win_flat_1),
                .win_flat_ch2(win_flat_2),
                .win_flat_ch3(win_flat_3),
                .win_flat_ch4(win_flat_4),
                .win_flat_ch5(win_flat_5),
                
                .valid_in(window_valid),
                
                .weight_flat(current_weights_flat), // Dây weight phẳng
                .bias(BIAS[i]), 
                
                .conv_out(pe_results[i]),
                .valid_out(pe_valid_flags[i])
            );
        end
    endgenerate

    // --- GÁN KẾT QUẢ RA CỔNG OUTPUT (MAPPING) ---
    always @(*) begin
        fmap_out_0  = pe_results[0];
        fmap_out_1  = pe_results[1];
        fmap_out_2  = pe_results[2];
        fmap_out_3  = pe_results[3];
        fmap_out_4  = pe_results[4];
        fmap_out_5  = pe_results[5];
        fmap_out_6  = pe_results[6];
        fmap_out_7  = pe_results[7];
        fmap_out_8  = pe_results[8];
        fmap_out_9  = pe_results[9];
        fmap_out_10 = pe_results[10];
        fmap_out_11 = pe_results[11];
        fmap_out_12 = pe_results[12];
        fmap_out_13 = pe_results[13];
        fmap_out_14 = pe_results[14];
        fmap_out_15 = pe_results[15];
    end

    // Valid Out: Chỉ cần lấy của PE 0 (vì tất cả chạy song song đồng bộ)
    assign valid_out = pe_valid_flags[0];
endmodule