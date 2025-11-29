import torch
import os

# Hàm chuyển đổi Hex 16-bit bù 2 sang số thực (Float)
def hex_to_float(hex_str):
    # Chuyển từ hex sang int 16-bit
    val_int = int(hex_str, 16)
    
    # Xử lý bù 2 (nếu bit dấu (bit 15) là 1 thì là số âm)
    if val_int & 0x8000:
        val_int = val_int - 0x10000
        
    # Chia cho 256 để ra số thực (Fixed-point Q8.8)
    return val_int / 256.0

def verify_weights(model_weight_tensor, hex_filename):
    print(f"--- Verifying {hex_filename} ---")
    
    # 1. Lấy dữ liệu gốc từ PyTorch (Làm phẳng để dễ so sánh)
    original_data = model_weight_tensor.detach().numpy().flatten()
    
    # 2. Đọc dữ liệu từ file Hex
    with open(hex_filename, 'r') as f:
        hex_lines = f.readlines()
        
    # 3. So sánh
    if len(original_data) != len(hex_lines):
        print(f"❌ ERROR: Size mismatch! Model: {len(original_data)}, File: {len(hex_lines)}")
        return

    max_error = 0.0
    for i, (orig, line) in enumerate(zip(original_data, hex_lines)):
        decoded_val = hex_to_float(line.strip())
        error = abs(orig - decoded_val)
        
        # Cập nhật sai số lớn nhất
        if error > max_error: max_error = error
        
        # In thử 5 giá trị đầu tiên để kiểm tra mắt
        if i < 5:
            print(f"idx {i}: PyTorch={orig:.6f} | HexDecoded={decoded_val:.6f} | Hex={line.strip()}")

    # Đánh giá sai số (Quantization Error)
    # Vì Q8.8 có độ phân giải là 1/256 = 0.00390625
    # Nên sai số < 0.004 là chấp nhận được.
    print(f"✅ Max Quantization Error: {max_error:.6f}")
    if max_error < 0.004:
        print("=> PASS: File Hex chuẩn xác!")
    else:
        print("=> WARNING: Sai số hơi lớn, kiểm tra lại logic export.")
    print("-" * 30)

# --- CHẠY KIỂM TRA ---
# (Bạn cần load lại model hoặc chạy code này ngay sau đoạn train trong cùng 1 script)
# Giả sử 'model' là biến model bạn vừa train xong
# verify_weights(model.c1.weight, "verilog_data/c1_weight.hex")