import numpy as np

def compare_logs(hw_file, sw_file):
    print(f"Đang so sánh {hw_file} với {sw_file}...")
    
    try:
        hw_data = np.loadtxt(hw_file, dtype=int)
        sw_data = np.loadtxt(sw_file, dtype=int)
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    # Kiểm tra kích thước
    if hw_data.shape != sw_data.shape:
        print(f"❌ SAI KÍCH THƯỚC: HW {hw_data.shape} != SW {sw_data.shape}")
        # Thường HW sẽ ít dòng hơn SW một chút do độ trễ ban đầu, 
        # ta có thể cắt đuôi SW hoặc tìm vị trí khớp.
        # Nhưng với logic valid chuẩn, nó phải khớp 100 dòng (10x10).
        return

    # Tính sai số
    diff = hw_data - sw_data
    abs_diff = np.abs(diff)
    max_err = np.max(abs_diff)
    
    if max_err == 0:
        print("✅ TUYỆT VỜI! Kết quả khớp tuyệt đối (Bit-perfect match).")
    else:
        print(f"⚠️ CÓ SAI SỐ! Sai số lớn nhất: {max_err}")
        print("Các vị trí sai đầu tiên:")
        mismatch_indices = np.where(abs_diff > 0)
        for i in range(min(5, len(mismatch_indices[0]))):
            r, c = mismatch_indices[0][i], mismatch_indices[1][i]
            print(f"  Row {r}, Col {c} (Channel {c}): HW={hw_data[r,c]}, SW={sw_data[r,c]}")

# Gọi hàm
compare_logs("output_c3_log.txt", "golden_c3_ref.txt")