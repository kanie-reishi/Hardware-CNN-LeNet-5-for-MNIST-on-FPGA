import numpy as np

GOLDEN_FILE = "golden_c5_ref.txt"
LOG_FILE = "output_c5_log.txt"

def verify():
    print("🔍 Đang so sánh C5...")
    try:
        # Load output từ HW
        # Mỗi dòng là 1 ảnh (120 số)
        hw_data = np.loadtxt(LOG_FILE, dtype=int)
        
        # Load golden
        gold_data = np.loadtxt(GOLDEN_FILE, dtype=int)
        
        # Xử lý nếu file chỉ có 1 dòng (numpy loadtxt sẽ ra mảng 1 chiều)
        if hw_data.ndim == 1: hw_data = hw_data.reshape(1, -1)
        if gold_data.ndim == 1: gold_data = gold_data.reshape(1, -1)

    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # So sánh
    matches = 0
    errors = 0
    
    # Tự động lấy số lượng ảnh dựa trên dữ liệu (min của HW và Gold để tránh lỗi)
    num_imgs = min(hw_data.shape[0], gold_data.shape[0])
    print(f"ℹ️ Đang kiểm tra {num_imgs} ảnh (HW: {hw_data.shape[0]}, Gold: {gold_data.shape[0]})...")
    
    for i in range(num_imgs):
        diff = hw_data[i] - gold_data[i]
        abs_diff = np.abs(diff)
        
        if np.all(diff == 0):
            print(f"✅ Ảnh {i}: PASS (Match tuyệt đối)")
            matches += 1
        else:
            print(f"❌ Ảnh {i}: FAIL")
            print(f"   Max Diff: {np.max(abs_diff)}")
            # In ra vài vị trí sai
            err_indices = np.where(diff != 0)[0]
            for idx in err_indices[:5]:
                print(f"   Index {idx}: HW={hw_data[i][idx]}, Gold={gold_data[i][idx]}")
            errors += 1

    if errors == 0:
        print("\n🎉 CHÚC MỪNG! Layer C5 hoạt động hoàn hảo.")
    else:
        print(f"\n⚠️ Có {errors} ảnh bị sai lệch.")

if __name__ == "__main__":
    verify()