`timescale 1ns/1ps

module tb_conv_layer_c1;

    // ============================================================
    // 1. KHAI BÁO TÍN HIỆU
    // ============================================================
    reg clk;
    reg rst_n;
    reg [7:0] pixel_in;
    reg valid_in;

    // Output từ Layer C1
    wire signed [23:0] fmap_out_0;
    wire signed [23:0] fmap_out_1;
    wire signed [23:0] fmap_out_2;
    wire signed [23:0] fmap_out_3;
    wire signed [23:0] fmap_out_4;
    wire signed [23:0] fmap_out_5;
    wire valid_out;

    // Bộ nhớ để chứa ảnh test (Input Image)
    // Ảnh 32x32 = 1024 pixel
    reg [7:0] img_mem [0:1023];
    
    // Biến hỗ trợ chạy vòng lặp
    integer i;
    integer file_out_1; // File handle để ghi kết quả

    // ============================================================
    // 2. INSTANTIATE DUT (Device Under Test)
    // ============================================================
    conv_layer_c1 u_dut (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .fmap_out_0(fmap_out_0),
        .fmap_out_1(fmap_out_1),
        .fmap_out_2(fmap_out_2),
        .fmap_out_3(fmap_out_3),
        .fmap_out_4(fmap_out_4),
        .fmap_out_5(fmap_out_5),
        .valid_out(valid_out)
    );

    // ============================================================
    // 3. CLOCK GENERATION (10ns -> 100MHz)
    // ============================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ============================================================
    // 4. TEST PROCESS
    // ============================================================
    initial begin
        // --- A. Chuẩn bị file ---
        // Load ảnh từ file hex do Python tạo ra
        // Bạn cần chắc chắn file này nằm cùng thư mục mô phỏng
        $readmemh("test_img_0_label_7.hex", img_mem);
        
        // Mở file để ghi kết quả output (để debug)
        file_out_1 = $fopen("output_c1_log.txt", "w");

        // --- B. Khởi tạo ---
        $display("-------------------------------------------");
        $display("   STARTING TESTBENCH FOR CONV LAYER C1");
        $display("-------------------------------------------");
        
        rst_n = 0;
        pixel_in = 0;
        valid_in = 0;
        #100; // Giữ reset một lúc
        rst_n = 1;
        #20;

        // --- C. Bắn dữ liệu (Streaming Image) ---
        $display("[T=%0t] Start Streaming 32x32 Image...", $time);

        for (i = 0; i < 1024; i = i + 1) begin
            // Đưa pixel vào ở cạnh xuống (để setup time thoải mái cho cạnh lên)
            @(negedge clk); 
            pixel_in = img_mem[i];
            valid_in = 1;
            
            // (Tuỳ chọn) In ra tiến độ
            if (i % 128 == 0) $display("   Streaming pixel %d/1024...", i);
        end

        // Sau khi bắn hết ảnh, ngắt valid
        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;
        $display("[T=%0t] Streaming Done. Waiting for pipeline flush...", $time);

        // --- D. Chờ kết quả chạy nốt ra ngoài ---
        // Pipeline và Line Buffer còn trễ một chút, chờ khoảng 100 clock
        #1000;
        
        $display("-------------------------------------------");
        $display("   TEST FINISHED. CHECK output_c1_log.txt");
        $display("-------------------------------------------");
        $fclose(file_out_1);
        $finish;
    end

    // ============================================================
    // 5. MONITOR / CAPTURE OUTPUT
    // ============================================================
    // Logic: Bất cứ khi nào valid_out = 1, ta ghi lại giá trị
    
    always @(posedge clk) begin
        if (valid_out) begin
            // In ra Console (ít thôi kẻo loạn)
            // $display("Time: %0t | Out: %d %d %d ...", $time, fmap_out_0, fmap_out_1, ...);
            
            // Ghi vào file Log dạng CSV hoặc Text để dễ check
            // Format: CH0  CH1  CH2  CH3  CH4  CH5
            $fwrite(file_out_1, "%d %d %d %d %d %d\n", 
                    fmap_out_0, fmap_out_1, fmap_out_2, 
                    fmap_out_3, fmap_out_4, fmap_out_5);
        end
    end

endmodule