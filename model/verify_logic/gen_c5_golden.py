import numpy as np

# CẤU HÌNH
INPUT_S4_FILE = "output_s4_log.txt"   # Output từ lớp trước
WEIGHT_FILE = "c5_weight.hex"         # File hex trọng số
OUTPUT_GOLDEN = "golden_c5_ref.txt"   # Kết quả mong đợi

IN_CH = 16
OUT_CH = 120
IMG_SIZE = 5  # 5x5

def load_hex_weights():
    # Load weights: 120 filters, mỗi filter 5x5x16
    # Tổng: 48,000 số
    try:
        with open(WEIGHT_FILE, 'r') as f:
            # Đọc file hex (có thể có khoảng trắng hoặc xuống dòng)
            data = f.read().split()
            weights = [int(x, 16) if int(x, 16) < 128 else int(x, 16) - 256 for x in data]
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy {WEIGHT_FILE}. Tạo random để test.")
        weights = np.random.randint(-10, 10, 48000).tolist()
    
    # Reshape về (120 Output, 25 Pixel, 16 Input)
    # Lưu ý thứ tự này phải khớp với cách bạn tạo file hex ban đầu
    # Logic hardware: [Output][Pixel][Input]
    w_np = np.array(weights).reshape(OUT_CH, IMG_SIZE * IMG_SIZE, IN_CH)
    return w_np

def main():
    # 1. Load Input S4
    try:
        s4_data = np.loadtxt(INPUT_S4_FILE, dtype=int)
    except:
        print("❌ Không đọc được file log S4.")
        return

    # Xử lý trường hợp file log S4 chứa nhiều ảnh hoặc 1 ảnh
    # S4 log: Mỗi dòng 16 số. 1 Ảnh = 25 dòng.
    num_rows = s4_data.shape[0]
    num_imgs = num_rows // 25
    
    if num_imgs == 0:
        print("❌ Dữ liệu S4 không đủ 1 ảnh (cần 25 dòng).")
        return

    print(f"👉 Tìm thấy {num_imgs} ảnh input từ S4.")

    # 2. Load Weights
    weights = load_hex_weights() # Shape: (120, 25, 16)
    
    # 3. Tính toán
    golden_results = []
    
    for img_idx in range(num_imgs):
        # Lấy 25 dòng của ảnh hiện tại
        # Shape: (25, 16)
        start_row = img_idx * 25
        img_block = s4_data[start_row : start_row + 25, :]
        
        # Tính C5 (Fully Connected)
        # Công thức: Output[k] = Sum(Image * Weight[k]) + Bias
        # Image: (25, 16), Weight[k]: (25, 16) -> Element-wise mult -> Sum all
        
        img_out = []
        for out_k in range(OUT_CH):
            # Lấy bộ weight cho output k
            w_k = weights[out_k] # (25, 16)
            
            # Nhân và cộng tổng
            # Có thể bias ở đây (tạm thời = 0 theo code Verilog của bạn)
            val = np.sum(img_block * w_k)
            
            # Saturation (Mô phỏng 32-bit signed clamp)
            val = max(min(val, 2147483647), -2147483648)
            img_out.append(val)
            
        golden_results.append(img_out)

    # 4. Lưu file
    print(f"💾 Lưu kết quả vào {OUTPUT_GOLDEN}...")
    np.savetxt(OUTPUT_GOLDEN, np.array(golden_results), fmt='%d', delimiter=' ')
    print("✅ Done.")

if __name__ == "__main__":
    main()