import numpy as np

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
FEATURE_MAP_SIZE = 28
NUM_FILTERS = 6
LOG_FILE_VERILOG = "output_s2_log.txt"
GOLDEN_FILE = "golden_s2_ref.txt"

# ==========================================
# HÀM HỖ TRỢ
# ==========================================
def load_golden():
    """Đọc file golden thành mảng numpy (14, 14, 6)"""
    golden_data = []
    with open(GOLDEN_FILE, 'r') as f:
        for line in f:
            if 'x' in line: continue
            parts = [int(x) for x in line.strip().split()]
            if len(parts) == NUM_FILTERS:
                golden_data.append(parts)
    
    return np.array(golden_data).reshape((FEATURE_MAP_SIZE//2, FEATURE_MAP_SIZE//2, NUM_FILTERS))
def load_verilog_output():
    """Đọc file log Verilog thành mảng numpy (14, 14, 6)"""
    verilog_data = []
    with open(LOG_FILE_VERILOG, 'r') as f:
        for line in f:
            if 'x' in line: continue
            parts = [int(x) for x in line.strip().split()]
            if len(parts) == NUM_FILTERS:
                verilog_data.append(parts)
                
    expected_len = (FEATURE_MAP_SIZE // 2) ** 2
    if len(verilog_data) != expected_len:
        print(f"Warning: Dữ liệu Verilog có {len(verilog_data)} dòng (Kỳ vọng: {expected_len}).")
        if len(verilog_data) > expected_len: verilog_data = verilog_data[:expected_len]

    return np.array(verilog_data).reshape((FEATURE_MAP_SIZE//2, FEATURE_MAP_SIZE//2, NUM_FILTERS))

# ==========================================
# HÀM KIỂM TRA
# ==========================================

def verify_s2():
    golden = load_golden()
    verilog = load_verilog_output()
    print("Verifying S2 Output...")
    if np.array_equal(golden, verilog):
        print("S2 Verification PASSED: Verilog output matches Golden reference.")
    else:
        print("S2 Verification FAILED: Discrepancies found between Verilog output and Golden reference.")
        # Tìm và in vị trí sai
        diffs = np.where(golden != verilog)
        for r, c, f in zip(*diffs):
            print(f"Mismatch at (Row: {r}, Col: {c}, Filter: {f}): Golden={golden[r,c,f]}, Verilog={verilog[r,c,f]}")

# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    verify_s2()
