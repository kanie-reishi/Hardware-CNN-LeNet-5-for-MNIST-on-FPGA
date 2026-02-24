`timescale 1ns/1ps

module tb_lenet5_top;

    // --- PARAMETERS ---
    parameter IN_PIXEL_WIDTH = 8;
    parameter OUT_LOGITS_WIDTH = 32;
    parameter FRONTEND_OUT_WIDTH = 32;

    // --- SIGNALS ---
    reg clk;
    reg rst_n;
    reg valid_in;
    reg [IN_PIXEL_WIDTH-1:0] pixel_in;
    
    wire valid_out;
    wire [OUT_LOGITS_WIDTH*10-1:0] logits_out;

    // --- VARIABLES ---
    integer i, c;
    integer timeout_cnt;
    reg signed [OUT_LOGITS_WIDTH-1:0] final_logits [0:9];

    // --- INSTANTIATE DUT (Device Under Test) ---
    lenet5_top #(
        .IN_PIXEL_WIDTH(IN_PIXEL_WIDTH),
        .OUT_LOGITS_WIDTH(OUT_LOGITS_WIDTH),
        .FRONTEND_OUT_WIDTH(FRONTEND_OUT_WIDTH)
    ) u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .pixel_in(pixel_in),
        .valid_out(valid_out),
        .logits_out(logits_out)
    );

    // --- CLOCK GENERATION (100MHz) ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // --- MAIN TEST SCENARIO ---
    initial begin
        // 1. KHỞI TẠO TÍN HIỆU
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
        
        // Giữ reset một thời gian để toàn bộ pipeline và memory khởi tạo
        #100;
        rst_n = 1;
        #50;

        $display("==================================================");
        $display("🚀 BẮT ĐẦU TEST TOÀN HỆ THỐNG LENET-5 (WIRING CHECK)");
        $display("==================================================");

        // 2. BƠM 1024 PIXELS (Ảnh 32x32) VÀO MẠNG
        // Quá trình này mô phỏng luồng Streaming từ Camera hoặc DMA
        $display("[Time: %0t] Đang truyền ảnh giả (1024 pixels) vào Frontend...", $time);
        
        for (i = 0; i < 1024; i = i + 1) begin
            @(negedge clk);
            valid_in = 1;
            pixel_in = 1; // Truyền pixel giả (giá trị = 1)
        end
        
        // Dừng truyền
        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;
        $display("[Time: %0t] Đã truyền xong ảnh. Đang chờ Backend tính toán...", $time);

        // 3. CHỜ KẾT QUẢ TỪ BACKEND (Với cơ chế Timeout bảo vệ)
        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 20000) begin // Đợi tối đa 20,000 nhịp clock
            @(posedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        // 4. KIỂM TRA & HIỂN THỊ KẾT QUẢ
        if (timeout_cnt >= 20000) begin
            $display("❌ LỖI TIMEOUT: Không nhận được tín hiệu valid_out từ Backend!");
            $display("-> Gợi ý: Kiểm tra lại các dây valid nối giữa C1->S2->C3->S4->C5->F6->Out");
        end else begin
            $display("✅ THÀNH CÔNG: Đã nhận được tín hiệu valid_out tại Time: %0t", $time);
            $display("-> Đi dây Dataflow cơ bản đã chính xác!");
            
            // Giải nén bus 320-bit thành 10 giá trị rời rạc để dễ nhìn
            for (c = 0; c < 10; c = c + 1) begin
                final_logits[c] = logits_out[c*OUT_LOGITS_WIDTH +: OUT_LOGITS_WIDTH];
                $display("   - Lớp %0d: %d", c, final_logits[c]);
            end
            
            // Check trạng thái Unknown (X)
            if (logits_out === { (OUT_LOGITS_WIDTH*10) {1'bx} }) begin
                $display("⚠️ CẢNH BÁO: Đầu ra chứa giá trị X (Unknown).");
                $display("-> Lý do: Bạn chưa nạp file Weight/Bias hoặc đi dây Data bị đứt.");
            end
        end

        $display("==================================================");
        #200;
        $finish;
    end

endmodule