import numpy as np

GOLDEN_FILE = "golden_final_ref.txt"
LOG_FILE = "output_final_log.txt" # Đổi tên file này theo log của ModelSim/Vivado

def verify():
    print("🔍 Đang so sánh Output Layer...")
    try:
        # Load output từ HW
        hw_data = np.loadtxt(LOG_FILE, dtype=int)
        # Load golden
        gold_data = np.loadtxt(GOLDEN_FILE, dtype=int)
        
        # Reshape nếu cần (để đảm bảo là mảng 2 chiều: [số_ảnh, 10_lớp])
        if hw_data.ndim == 1: hw_data = hw_data.reshape(1, -1)
        if gold_data.ndim == 1: gold_data = gold_data.reshape(1, -1)

    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        print(f"   (Hãy chắc chắn bạn đã export log từ Verilog ra file '{LOG_FILE}')")
        return

    num_imgs = min(hw_data.shape[0], gold_data.shape[0])
    print(f"ℹ️ Đang kiểm tra {num_imgs} ảnh...")
    
    matches = 0
    errors = 0
    
    for i in range(num_imgs):
        # 1. So sánh giá trị tuyệt đối (Bit-exact check)
        diff = hw_data[i] - gold_data[i]
        is_exact_match = np.all(diff == 0)
        
        # 2. So sánh kết quả dự đoán (Argmax check)
        hw_pred = np.argmax(hw_data[i])
        gold_pred = np.argmax(gold_data[i])
        
        print(f"--- Ảnh {i} ---")
        print(f"   HW Pred: {hw_pred} | Gold Pred: {gold_pred}")
        
        if is_exact_match:
            print(f"   ✅ PASS: Giá trị khớp tuyệt đối.")
            matches += 1
        else:
            print(f"   ❌ FAIL: Giá trị bị lệch.")
            print(f"      Max Diff: {np.max(np.abs(diff))}")
            if hw_pred == gold_pred:
                print(f"      ⚠️ Tuy nhiên, kết quả dự đoán vẫn ĐÚNG (Cùng ra số {hw_pred}).")
            else:
                print(f"      ⛔ SAI KẾT QUẢ DỰ ĐOÁN (HW ra {hw_pred} thay vì {gold_pred}).")
            errors += 1

    print(f"\nKẾT QUẢ: {matches}/{num_imgs} ảnh khớp hoàn toàn.")

if __name__ == "__main__":
    verify()
