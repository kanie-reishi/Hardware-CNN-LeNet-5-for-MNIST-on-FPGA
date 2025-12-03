module line_buffer_shift (
    input wire clk,
    input wire rst_n,       
    input [7:0] data_in, 
    input wire valid_in, 
    
    output reg [7:0] dout_line1,
    output reg [7:0] dout_line2, 
    output reg [7:0] dout_line3, 
    output reg [7:0] dout_line4,

    output wire valid_out 
);
    parameter IMG_WIDTH = 32;
    localparam LB_DEPTH = 32; 

    // Bộ nhớ Line Buffer
    reg [7:0] line_buffer0 [0:LB_DEPTH-1];
    reg [7:0] line_buffer1 [0:LB_DEPTH-1];
    reg [7:0] line_buffer2 [0:LB_DEPTH-1];
    reg [7:0] line_buffer3 [0:LB_DEPTH-1];

    integer i;
    reg [4:0] x_cnt;
    reg [4:0] y_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset bộ đếm
            x_cnt <= 0;
            y_cnt <= 0;
            // Reset output (tùy chọn)
            dout_line1 <= 0; dout_line2 <= 0; dout_line3 <= 0; dout_line4 <= 0;
        end 
        else if (valid_in) begin
            // A. Update coordinate counters
            if(x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                if(y_cnt == IMG_WIDTH - 1) y_cnt <= 0;
                else y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end

            // B. Shift Logic (Logic nối đuôi)
            // Lấy ra từ cuối Buffer 3 (Dòng cũ nhất)
            dout_line4 <= line_buffer3[LB_DEPTH-1];
            
            // Dịch chuyển Buffer 3
            line_buffer3[0] <= dout_line3; // Nối với đầu ra Buffer 2
            for(i = 1; i < LB_DEPTH; i = i + 1) line_buffer3[i] <= line_buffer3[i-1];

            // Buffer 2 -> Buffer 3
            dout_line3 <= line_buffer2[LB_DEPTH-1];
            line_buffer2[0] <= dout_line2;
            for(i = 1; i < LB_DEPTH; i = i + 1) line_buffer2[i] <= line_buffer2[i-1];

            // Buffer 1 -> Buffer 2
            dout_line2 <= line_buffer1[LB_DEPTH-1];
            line_buffer1[0] <= dout_line1;
            for(i = 1; i < LB_DEPTH; i = i + 1) line_buffer1[i] <= line_buffer1[i-1];

            // Buffer 0 -> Buffer 1
            dout_line1 <= line_buffer0[LB_DEPTH-1];
            line_buffer0[0] <= data_in; // Nạp pixel mới nhất vào đầu Buffer 0
            for(i = 1; i < LB_DEPTH; i = i + 1) line_buffer0[i] <= line_buffer0[i-1];
        end
    end

    // Valid logic (Bỏ qua 4 dòng đầu và 4 cột đầu - Padding Mask)
    assign valid_out = (x_cnt >= 4 && y_cnt >= 4) && valid_in;

endmodule