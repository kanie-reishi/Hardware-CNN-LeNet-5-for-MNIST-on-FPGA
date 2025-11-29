module line_buffer_5lines #(
    parameter IMG_WIDTH = 32
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,
    input  wire        valid_in,

    output wire [7:0]  line0_out, // Dòng hiện tại (Pass-through)
    output reg  [7:0]  line1_out, // Trễ 1 dòng
    output reg  [7:0]  line2_out, // Trễ 2 dòng
    output reg  [7:0]  line3_out, // Trễ 3 dòng
    output reg  [7:0]  line4_out, // Trễ 4 dòng

    output wire        valid_out
);

    reg [4:0] x_cnt;
    reg [4:0] y_cnt;

    // Ép kiểu Distributed RAM (LUTRAM) để tiết kiệm BRAM cho mạng nhỏ
    (* ram_style = "distributed" *) reg [7:0] lb0 [0:IMG_WIDTH-1];
    (* ram_style = "distributed" *) reg [7:0] lb1 [0:IMG_WIDTH-1];
    (* ram_style = "distributed" *) reg [7:0] lb2 [0:IMG_WIDTH-1];
    (* ram_style = "distributed" *) reg [7:0] lb3 [0:IMG_WIDTH-1];

    // 1. Line 0 là dòng đang đi vào (không cần lưu, dùng ngay)
    assign line0_out = data_in;

    always @(posedge clk) begin
        if (!rst_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
            // Reset output (Tùy chọn, tốt cho debug)
            line1_out <= 0; line2_out <= 0; line3_out <= 0; line4_out <= 0;
        end 
        else if (valid_in) begin
            // --- CƠ CHẾ SHIFT (Dựa vào tính chất non-blocking <=) ---
            
            // 1. Đọc dữ liệu ra từ các bộ nhớ (Lấy giá trị CŨ)
            // Vì dùng Distributed RAM (hoạt động như Register), ta đọc trực tiếp được
            line1_out <= lb0[x_cnt];
            line2_out <= lb1[x_cnt];
            line3_out <= lb2[x_cnt];
            line4_out <= lb3[x_cnt];

            // 2. Dịch chuyển dữ liệu giữa các Line Buffer
            // lb1 nhận dữ liệu cũ của lb0 tại cùng vị trí cột x_cnt
            lb1[x_cnt] <= lb0[x_cnt]; 
            lb2[x_cnt] <= lb1[x_cnt];
            lb3[x_cnt] <= lb2[x_cnt];
            
            // 3. Nạp dữ liệu mới vào Line Buffer đầu tiên
            lb0[x_cnt] <= data_in;

            // --- BỘ ĐẾM TỌA ĐỘ ---
            if (x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                if (y_cnt == IMG_WIDTH - 1) begin
                    y_cnt <= 0;
                end else begin
                    y_cnt <= y_cnt + 1;
                end
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end
    end

    // Valid Logic: Bỏ qua 4 dòng đầu và 4 cột đầu mỗi dòng (Padding mask)
    assign valid_out = valid_in && (x_cnt >= 4) && (y_cnt >= 4);

endmodule