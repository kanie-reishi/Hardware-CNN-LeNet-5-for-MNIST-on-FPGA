import numpy as np
import os

# --- CẤU HÌNH ---
INPUT_FILE = "output_s2_log.txt"    # Input từ lớp S2
WEIGHT_FILE = "c3_weight.hex"       # File chứa 2400 trọng số (8-bit)
BIAS_FILE = "c3_bias.hex"           # File chứa 16 bias (nếu không có sẽ tự tạo 0)
OUTPUT_FILE = "golden_c3_ref.txt"   # File kết quả để so sánh

# Kích thước mạng
IN_H, IN_W = 14, 14
IN_CH = 6
OUT_CH = 16
KERNEL_SIZE = 5
STRIDE = 1
OUT_H = (IN_H - KERNEL_SIZE) // STRIDE + 1  # = 10
OUT_W = (IN_W - KERNEL_SIZE) // STRIDE + 1  # = 10

def load_hex_weights(filename, count):
    """Đọc file hex và trả về mảng numpy. Hỗ trợ cả định dạng hex dính liền hoặc cách dòng."""
    weights = []
    try:
        with open(filename, 'r') as f:
            content = f.read().split() # Tách theo khoảng trắng hoặc xuống dòng
            for hex_str in content:
                # Xử lý số signed 8-bit (nếu file hex là 2 ký tự)
                val = int(hex_str, 16)
                if val > 127: val -= 256 # Convert to signed 8-bit
                weights.append(val)
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy {filename}. Đang tạo ngẫu nhiên để test...")
        return np.random.randint(-10, 10, count)
    
    if len(weights) < count:
        print(f"⚠️ Cảnh báo: File {filename} chỉ có {len(weights)} số (cần {count}). Đang đệm thêm 0.")
        weights += [0] * (count - len(weights))
    
    return np.array(weights[:count])

def load_s2_input(filename):
    """Đọc log output từ lớp S2"""
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Bỏ qua dòng chứa 'x' hoặc dòng trống
                if 'x' in line or not line.strip(): continue
                parts = [int(x) for x in line.strip().split()]
                if len(parts) == IN_CH:
                    data.append(parts)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy {filename}. Hãy chạy mô phỏng S2 trước.")
        exit()

    # Reshape về (14, 14, 6)
    # Lưu ý: data là list các dòng (196 dòng), mỗi dòng 6 kênh
    np_data = np.array(data)
    if np_data.shape[0] != IN_H * IN_W:
        print(f"⚠️ Cảnh báo: Input S2 có {np_data.shape[0]} dòng (kỳ vọng {IN_H*IN_W}).")
        # Cắt hoặc padding nếu cần
        if np_data.shape[0] > IN_H * IN_W:
            np_data = np_data[:IN_H*IN_W]
        else:
            print("❌ Dữ liệu không đủ để chạy C3.")
            exit()
            
    return np_data.reshape(IN_H, IN_W, IN_CH)

def conv2d_fully_connected(fmap_in, weights, biases):
    """
    Mô phỏng chính xác FPGA Fully Connected Conv
    fmap_in: (14, 14, 6)
    weights: (16, 6, 5, 5)
    biases: (16,)
    """
    fmap_out = np.zeros((OUT_H, OUT_W, OUT_CH), dtype=int)
    
    print(f"👉 Bắt đầu tính toán Convolution C3...")
    print(f"   Input: {fmap_in.shape}, Output mong muốn: {fmap_out.shape}")

    # Giới hạn 32-bit signed cho Saturation
    MAX_INT32 = 2147483647
    MIN_INT32 = -2147483648

    # Loop qua từng pixel output (y, x)
    for r in range(OUT_H):
        for c in range(OUT_W):
            # Loop qua từng Output Channel (Filter) - Tương ứng 16 PE song song
            for o in range(OUT_CH):
                
                accumulator = biases[o] # Khởi tạo với Bias
                
                # Loop qua từng Input Channel - Tương ứng việc cộng gộp 6 kênh
                for i in range(IN_CH):
                    # Cắt cửa sổ 5x5 từ Input channel i
                    # Tương ứng Line Buffer + Window Array
                    window = fmap_in[r:r+KERNEL_SIZE, c:c+KERNEL_SIZE, i]
                    
                    # Lấy Kernel tương ứng
                    kernel = weights[o, i, :, :]
                    
                    # Nhân chập và cộng dồn
                    # Đây là phép toán sum(A * B)
                    conv_sum = np.sum(window * kernel)
                    
                    accumulator += conv_sum
                
                # --- FIX: Mô phỏng Saturation của Hardware ---
                # Nếu giá trị vượt quá 32-bit, kẹp nó lại giống hệt Verilog
                if accumulator > MAX_INT32:
                    accumulator = MAX_INT32
                elif accumulator < MIN_INT32:
                    accumulator = MIN_INT32
                
                fmap_out[r, c, o] = accumulator

    return fmap_out

def main():
    # 1. Load Input S2
    s2_data = load_s2_input(INPUT_FILE)
    
    # 2. Load Weights
    # Cần 2400 weights. Shape mong muốn: (Out, In, K, K) = (16, 6, 5, 5)
    raw_weights = load_hex_weights(WEIGHT_FILE, 2400)
    # Reshape theo thứ tự mà FPGA đọc: 
    # Thường file hex lưu: [Out0_In0], [Out0_In1]... [Out0_In5], [Out1_In0]...
    weights_c3 = raw_weights.reshape(OUT_CH, IN_CH, KERNEL_SIZE, KERNEL_SIZE)
    
    # 3. Load Biases
    # Nếu chưa có file bias, tạo file tạm toàn số 0
    if not os.path.exists(BIAS_FILE):
        print(f"ℹ️ Không thấy {BIAS_FILE}, tạo bias = 0.")
        with open(BIAS_FILE, 'w') as f:
            f.write("00 " * 16)
    raw_biases = load_hex_weights(BIAS_FILE, 16)
    
    # 4. Tính toán
    c3_output = conv2d_fully_connected(s2_data, weights_c3, raw_biases)
    
    # 5. Xuất file Golden Data
    print(f"💾 Đang lưu kết quả vào {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        # Flatten về dạng: Mỗi dòng là 1 pixel (có 16 kênh)
        # Shape (10, 10, 16) -> (100, 16)
        flat_out = c3_output.reshape(-1, OUT_CH)
        for row in flat_out:
            line = " ".join(map(str, row))
            f.write(line + "\n")
            
    print("✅ Hoàn tất! Sẵn sàng verify.")

if __name__ == "__main__":
    main()