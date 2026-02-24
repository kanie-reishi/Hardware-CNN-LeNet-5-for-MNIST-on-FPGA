import torch
import numpy as np

# --- CẤU HÌNH ---
OUTPUT_HEX_FILE = "output_weight.hex"
OUTPUT_BIAS_FILE = "output_bias.hex"
SCALE_FACTOR = 64.0  # Hệ số Quantization (Ví dụ: Q2.6)

# Kích thước Output Layer
NUM_ROWS = 10  # 10 Neurons (Output Classes 0-9)
NUM_COLS = 84  # 84 Inputs từ F6

def float_to_int8_hex(val_float):
    """Chuyển float sang int8 hex (Two's complement)"""
    # 1. Scale & Round
    val_int = int(round(val_float * SCALE_FACTOR))
    
    # 2. Clamp về [-128, 127]
    val_int = max(min(val_int, 127), -128)
    
    # 3. Convert sang unsigned 8-bit cho Hex
    val_u8 = val_int & 0xFF
    return f"{val_u8:02X}"

def main():
    print(f"🔄 Đang tạo weight cho Output Layer ({NUM_ROWS}x{NUM_COLS})...")

    # 1. TẠO WEIGHT GIẢ LẬP (Hoặc load từ file .pth)
    # Ở đây mình tạo random để test, nếu bạn có model thật, hãy load state_dict
    torch.manual_seed(42) # Cố định seed để kết quả giống nhau mỗi lần chạy
    
    # Shape chuẩn của PyTorch Linear(84, 10) là (10, 84) -> (Out, In)
    # [Row 0]: Weight của Neuron 0 nối với 84 input
    # ...
    weights = torch.randn(NUM_ROWS, NUM_COLS) 
    biases = torch.randn(NUM_ROWS)

    # 2. XUẤT FILE WEIGHT (Format: 10 dòng, mỗi dòng 84 số)
    with open(OUTPUT_HEX_FILE, "w") as f:
        for r in range(NUM_ROWS):
            line_hex = []
            for c in range(NUM_COLS):
                # Lấy giá trị w[r][c]
                val = weights[r, c].item()
                hex_str = float_to_int8_hex(val)
                line_hex.append(hex_str)
            
            # Ghi cả dòng (Neuron r) vào file, cách nhau bởi dấu cách
            f.write(" ".join(line_hex) + "\n")
    
    print(f"✅ Đã lưu '{OUTPUT_HEX_FILE}' (10 lines x 84 bytes).")

    # 3. XUẤT FILE BIAS (Format: 10 dòng, mỗi dòng 1 số 32-bit)
    # Bias layer cuối cần độ chính xác cao hơn weight
    with open(OUTPUT_BIAS_FILE, "w") as f:
        for r in range(NUM_ROWS):
            val = biases[r].item()
            # Scale bias (thường scale = scale_input * scale_weight)
            # Ở đây giả sử scale giống weight để đơn giản hóa
            val_int = int(round(val * SCALE_FACTOR * SCALE_FACTOR)) 
            
            # 32-bit Hex
            val_u32 = val_int & 0xFFFFFFFF
            f.write(f"{val_u32:08X}\n")

    print(f"✅ Đã lưu '{OUTPUT_BIAS_FILE}'.")
    
    # --- KIỂM TRA LẠI (VERIFICATION) ---
    print("\n--- Preview File Hex ---")
    with open(OUTPUT_HEX_FILE, 'r') as f:
        lines = f.readlines()
        print(f"Số dòng thực tế: {len(lines)}")
        first_line_items = len(lines[0].strip().split())
        print(f"Số lượng weight ở dòng 0: {first_line_items}")
        
        if len(lines) == 10 and first_line_items == 84:
            print("🎉 FORMAT CHUẨN: Verilog sẽ đọc đúng vào mảng [0:9][0:83]!")
        else:
            print("⚠️ FORMAT SAI!")

if __name__ == "__main__":
    main()