import numpy as np

# CẤU HÌNH TÊN FILE
GOLDEN_FILE = "golden_s4_ref.txt"
VERILOG_FILE = "output_s4_log.txt"
WIDTH = 5
HEIGHT = 5
CHANNELS = 16

def verify_s4():
    print(f"🔍 Đang so sánh {VERILOG_FILE} với {GOLDEN_FILE}...")
    
    # 1. Đọc dữ liệu
    try:
        golden_data = np.loadtxt(GOLDEN_FILE, dtype=int)
        hw_data = np.loadtxt(VERILOG_FILE, dtype=int)
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    # 2. Kiểm tra kích thước
    # Golden Data có thể ngắn hơn hoặc dài hơn do cách tạo
    # HW Data thường có độ trễ ban đầu (toàn số 0 hoặc rác)
    
    # Chiến thuật: Tìm vị trí bắt đầu khớp nhau (Sync)
    # Vì output S4 là 5x5 = 25 dòng. Ta sẽ trượt để tìm vị trí khớp nhất.
    
    rows_gold = golden_data.shape[0]
    rows_hw = hw_data.shape[0]
    
    if rows_gold != rows_hw:
        print(f"⚠️ Cảnh báo: Số dòng lệch nhau (Gold: {rows_gold}, HW: {rows_hw}).")
        print("   -> Đang cố gắng tự động đồng bộ (Auto-sync)...")
    
    # Chuyển về dạng 3D để dễ hình dung (N dòng, 16 kênh) -> (H, W, 16)
    # Lưu ý: S4 output 5x5 = 25 vector.
    
    # Tìm điểm bắt đầu của dữ liệu thực trong file HW (Bỏ qua các dòng 0 ở đầu nếu có)
    # Giả sử dòng đầu tiên khác 0 của Golden là key để tìm
    
    start_idx_hw = -1
    # Lấy mẫu dòng đầu tiên của Golden (hoặc dòng thứ 5 cho chắc)
    sample_row = golden_data[0] 
    
    # Quét trong HW để tìm dòng này
    for i in range(rows_hw - 5):
        if np.array_equal(hw_data[i], sample_row):
            start_idx_hw = i
            break
            
    if start_idx_hw == -1:
        print("❌ KHÔNG TÌM THẤY điểm đồng bộ! Dữ liệu có vẻ sai lệch hoàn toàn.")
        print("   HW Row 0:", hw_data[0])
        print("   Gold Row 0:", golden_data[0])
        return
    else:
        print(f"✅ Đã tìm thấy điểm đồng bộ tại dòng {start_idx_hw} của file HW.")

    # 3. So sánh chi tiết từng pixel
    match_count = 0
    mismatch_count = 0
    
    print("\n--- KẾT QUẢ CHI TIẾT ---")
    
    # Chỉ so sánh số lượng dòng tương ứng (25 dòng cho 1 ảnh)
    check_len = min(rows_gold, rows_hw - start_idx_hw)
    
    for i in range(check_len):
        row_gold = golden_data[i]
        row_hw = hw_data[start_idx_hw + i]
        
        diff = row_gold - row_hw
        if np.any(diff != 0):
            mismatch_count += 1
            print(f"❌ Mismatch tại dòng {i} (HW Line {start_idx_hw + i}):")
            for c in range(CHANNELS):
                if row_gold[c] != row_hw[c]:
                    print(f"   Chan {c}: Gold={row_gold[c]}, HW={row_hw[c]}, Diff={diff[c]}")
            if mismatch_count > 5: # Chỉ in 5 lỗi đầu tiên
                print("... (Dừng in lỗi)")
                break
        else:
            match_count += 1

    if mismatch_count == 0:
        print(f"\n🎉 CHÚC MỪNG! Kết quả KHỚP TUYỆT ĐỐI trên {match_count} dòng.")
        print("   Layer S4 của bạn đã hoạt động chính xác 100%.")
    else:
        print(f"\n⚠️ CÓ LỖI: {mismatch_count} dòng bị sai lệch.")

if __name__ == "__main__":
    verify_s4()