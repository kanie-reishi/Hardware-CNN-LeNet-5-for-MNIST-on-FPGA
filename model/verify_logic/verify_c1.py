import numpy as np

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
IMG_WIDTH = 32
KERNEL_SIZE = 5
NUM_FILTERS = 6
OUTPUT_SIZE = 28 # 32 - 5 + 1
HEX_IMG_FILE = "test_img_0_label_7.hex"
HEX_WEIGHT_FILE = "c1_weight.hex"
LOG_FILE_VERILOG = "output_c1_log.txt"

# ==========================================
# HÀM HỖ TRỢ
# ==========================================
def to_signed_8bit(hex_str):
    val = int(hex_str, 16)
    if val >= 0x80:
        val -= 0x100
    return val

def load_image():
    """Đọc file hex ảnh thành ma trận numpy 32x32"""
    img = []
    with open(HEX_IMG_FILE, 'r') as f:
        for line in f:
            if line.startswith("//"): continue
            img.append(int(line.strip(), 16)) # Ảnh input là unsigned (0-255)
    return np.array(img).reshape((IMG_WIDTH, IMG_WIDTH))

def load_weights():
    """Đọc file hex weight thành mảng (6, 5, 5)"""
    weights = []
    with open(HEX_WEIGHT_FILE, 'r') as f:
        temp_w = []
        for line in f:
            temp_w.append(to_signed_8bit(line.strip()))
            
    # Reshape: 6 filters, mỗi filter 5x5
    return np.array(temp_w).reshape((NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE))

def software_convolution(img, weights):
    """Tính Conv2D giả lập phần cứng (Integer Math Only)"""
    # Output shape: (28, 28, 6)
    ref_output = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, NUM_FILTERS), dtype=int)
    
    print("Computing Software Convolution...")
    # Quét từng vị trí kernel (Sliding Window)
    for r in range(OUTPUT_SIZE):     # Dòng 0..27
        for c in range(OUTPUT_SIZE): # Cột 0..27
            # Cắt vùng ảnh 5x5
            window = img[r:r+KERNEL_SIZE, c:c+KERNEL_SIZE]
            
            # Tính cho 6 filters song song
            for f in range(NUM_FILTERS):
                # Nhân chập: Tổng (Window * Kernel) + Bias (Ở đây Bias=0)
                # Lưu ý: Phần cứng làm tròn số nguyên, ở đây Python int cũng vậy
                conv_sum = np.sum(window * weights[f])
                ref_output[r, c, f] = conv_sum
                
    return ref_output

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Tính toán chuẩn bằng Python
        img = load_image()
        weights = load_weights()
        sw_result = software_convolution(img, weights)
        
        # 2. Đọc kết quả từ ModelSim Log
        hw_results = []
        with open(LOG_FILE_VERILOG, 'r') as f:
            for line in f:
                # File log format: "val0 val1 val2 val3 val4 val5"
                vals = [int(x) for x in line.strip().split()]
                hw_results.append(vals)
        
        hw_results = np.array(hw_results) # Shape mong muốn: (784, 6)
        
        # 3. So sánh
        print(f"\n--- VERIFICATION REPORT ---")
        print(f"Software Result Shape: {sw_result.reshape(-1, 6).shape}")
        print(f"Hardware Result Shape: {hw_results.shape}")
        
        # Flatten SW result để so sánh từng dòng
        sw_flat = sw_result.reshape(-1, NUM_FILTERS)
        
        if hw_results.shape != sw_flat.shape:
            print("ERROR: Kích thước file Log không khớp với tính toán lý thuyết!")
            print(f"Kiểm tra lại thời gian chạy mô phỏng (Testbench).")
        else:
            diff = sw_flat - hw_results
            total_errors = np.count_nonzero(diff)
            
            if total_errors == 0:
                print("\n✅ PASSED! Kết quả phần cứng trùng khớp 100% với Python.")
            else:
                print(f"\n❌ FAILED! Có {total_errors} vị trí sai lệch.")
                print("Dòng đầu tiên bị sai:")
                for i in range(len(diff)):
                    if np.any(diff[i] != 0):
                        print(f"Index {i}: HW={hw_results[i]} | SW={sw_flat[i]}")
                        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file {e.filename}. Hãy copy file vào cùng thư mục script.")