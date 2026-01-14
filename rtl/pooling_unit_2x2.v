module pooling_unit_2x2 #(
    parameter WIDTH = 24, // Bit-width of input feature map pixel from Conv Layer
    parameter IMG_WIDTH = 28 // C1 output feature map width
)(
    input clk, 
    input rst_n,
    // Input feature map pixel stream
    input signed [WIDTH-1:0] feat_map_in,
    input valid_in,
    // Output pooled feature map pixel stream
    output reg signed [WIDTH-1:0] data_out,
    output reg valid_out
);
    // 1. Line Buffer to store one row of input feature map
    reg signed [WIDTH-1:0] line_buffer [0:IMG_WIDTH-1];
    wire signed [WIDTH-1:0] val_row_up; // Pixel at previous row, same column. Read from Line Buffer

    // 2. Registers to hold the left column pixel (Col N - 1)
    reg signed [WIDTH-1:0] val_row_curr_prev; // Pixel at same row, previous column
    reg signed [WIDTH-1:0] val_row_up_prev;  // Pixel at previous row, previous column

    // 3. Counters Register for Stride 2
    reg [4:0] x_cnt; // Column counter
    reg [4:0] y_cnt; // Row counter

    // -- LOGIC LINE BUFFER --
    always @(posedge clk or negedge rst_n) begin
            if(valid_in) begin
                // Update line buffer
                line_buffer[x_cnt] <= feat_map_in;
            end
        end
    assign val_row_up = line_buffer[x_cnt];
    // -- LOGIC WINDOW ARRAY & COMPARATOR FOR POOLING 2x2 --
    wire signed[WIDTH-1:0] p00, p01, p10, p11; // 4 pixels in 2x2 window

    // Mapping 2x2 window pixels
    // p00 p01 -> Previous Row
    // p10 p11 -> Current Row

    assign p11 = feat_map_in;
    assign p10 = val_row_curr_prev;
    assign p01 = val_row_up;
    assign p00 = val_row_up_prev;

    // Find Max (Combinational Logic)
    wire signed [WIDTH-1:0] max_top = (p00 > p01) ? p00 : p01;
    wire signed [WIDTH-1:0] max_bot = (p10 > p11) ? p10 : p11;
    wire signed [WIDTH-1:0] max_val = (max_top > max_bot) ? max_top : max_bot;

    // --- Main Sequential Logic ---
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            // Reset all registers counters and outputs
            x_cnt <= 0;
            y_cnt <= 0;
            val_row_curr_prev <= 0;
            val_row_up_prev <= 0;
            data_out <= 0;
            valid_out <= 0;
        end else if(valid_in) begin
            // A. Update coordinate counters
            if(x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                if(y_cnt == IMG_WIDTH - 1) y_cnt <= 0;
                else y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end

            // B. Update left column registers
            val_row_curr_prev <= feat_map_in; // Current row, previous column
            val_row_up_prev <= val_row_up;    // Previous row, previous column

            // C. Output max value at stride 2 positions
            // Output only when both x_cnt and y_cnt are odd (stride 2)
            if((x_cnt[0] == 1'b1) && (y_cnt[0] == 1'b1)) begin
                data_out <= max_val;
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule