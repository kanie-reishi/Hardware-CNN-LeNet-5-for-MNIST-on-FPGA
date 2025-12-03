`timescale 1ns/1ps

module tb_line_buffer_shift;

    // 1. Khai báo tín hiệu
    reg clk;
    reg rst_n;
    reg [7:0] data_in;
    reg valid_in;

    wire [7:0] dout_line1;
    wire [7:0] dout_line2;
    wire [7:0] dout_line3;
    wire [7:0] dout_line4;
    wire valid_out;

    // 2. Gọi DUT (Device Under Test)
    line_buffer_shift #(
        .IMG_WIDTH(32)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .data_in(data_in),
        .valid_in(valid_in),
        .dout_line1(dout_line1),
        .dout_line2(dout_line2),
        .dout_line3(dout_line3),
        .dout_line4(dout_line4),
        .valid_out(valid_out)
    );

    // 3. Tạo Clock (10ns chu kỳ -> 100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 4. Khai báo bộ nhớ chưa ảnh
    // Ảnh 32x32, mỗi pixel 8 bit
    reg [7:0] image_mem [0:1023]; // 32*32 = 1024 pixels
    integer i; // Biến chạy vòng lặp
    
    // Khối nạp file hex vào bộ nhớ
    initial begin

        $readmemh("model/verilog_data/test_img_0_label_7.hex", image_mem);

    end
    // --------------------------------
    initial begin
        // Setup Waveform
        $dumpfile("dump.vcd");
        $dumpvars(0, tb_line_buffer_shift);

        // A. Reset hệ thống
        rst_n = 0;
        valid_in = 0;
        data_in = 0;
        #20;
        rst_n = 1;
        #10;

        $display("--- Bat dau nap anh 32x32 ---");

        // B. Nạp ảnh 
        // Nạp từng pixel một, mỗi pixel 10ns
        for(i = 0; i < 1024; i = i + 1) begin
            @(posedge clk);
            valid_in = 1;
            data_in = image_mem[i];
        end

        // C. Kết thúc nạp
        @(posedge clk);
        valid_in = 0;
        data_in = 0;
        
        #200; // Chờ thêm một thời gian xử lý pipeline
        $display("--- Ket thuc mo phong ---");
        $finish;
    end

    // 5. Monitor kết quả (Tự động in ra console khi có Valid)
    always @(posedge clk) begin
        if (valid_out) begin
            $display("Time: %0t | Valid! | In(Row%0d): %h | L1: %h | L2: %h | L3: %h | L4: %h",
                     $time, uut.y_cnt, data_in, dout_line1, dout_line2, dout_line3, dout_line4);
                     
            // Check nhanh logic
            // Nếu data_in là dòng N, thì L1 phải là N-1, L2 là N-2...
            if ((data_in - 1 == dout_line1) && (dout_line1 - 1 == dout_line2))
                $display("    -> Logic Check: PASS (Dung cot, dung tre dong)");
            else
                $display("    -> Logic Check: FAIL");
        end
    end

endmodule