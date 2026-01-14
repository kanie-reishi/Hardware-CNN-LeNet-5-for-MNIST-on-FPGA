import numpy as np

# CẤU HÌNH
INPUT_FILE = "output_c1_log.txt"   # File log từ C1 (28x28)
GOLDEN_FILE = "golden_s2_ref.txt"  # File đáp án cho S2 (14x14)
INPUT_SIZE = 28
OUTPUT_SIZE = 14
CHANNELS = 6

def max_pooling_software():
    print(f"Reading {INPUT_FILE}...")
    
    # 1. Đọc dữ liệu từ log C1
    raw_data = []
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                # Bỏ qua dòng chứa 'x' nếu có (dòng rác đầu tiên)
                if 'x' in line: continue
                
                try:
                    parts = [int(x) for x in line.strip().split()]
                    if len(parts) == CHANNELS:
                        raw_data.append(parts)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {INPUT_FILE}. Hãy chạy mô phỏng C1 trước.")
        return

    # Chuyển thành numpy array (28, 28, 6)
    # Lưu ý: raw_data là list các dòng (784 dòng), mỗi dòng 6 kênh
    # Ta cần reshape lại đúng hình dạng ảnh
    expected_len = INPUT_SIZE * INPUT_SIZE
    if len(raw_data) != expected_len:
        print(f"Warning: Số lượng dòng ({len(raw_data)}) không khớp 28x28 ({expected_len}).")
        if len(raw_data) > expected_len:
            raw_data = raw_data[:expected_len]
        else:
            # Pad thêm số 0 nếu thiếu dữ liệu để tránh lỗi reshape
            missing = expected_len - len(raw_data)
            print(f" -> Tự động thêm {missing} dòng 0 để tiếp tục.")
            raw_data.extend([[0]*CHANNELS for _ in range(missing)])
        
    c1_fmaps = np.array(raw_data).reshape(INPUT_SIZE, INPUT_SIZE, CHANNELS)
    
    # 2. Thực hiện Max Pooling (2x2, Stride 2)
    s2_output = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, CHANNELS), dtype=int)
    
    print("Computing Max Pooling 2x2...")
    for r in range(OUTPUT_SIZE):      # 0..13
        for c in range(OUTPUT_SIZE):  # 0..13
            # Vùng 2x2 tương ứng trên Input
            r_start = r * 2
            c_start = c * 2
            
            # Cắt cửa sổ 2x2 cho tất cả 6 kênh cùng lúc
            window = c1_fmaps[r_start:r_start+2, c_start:c_start+2, :]
            # Tìm Max dọc theo trục 0 và 1 (chiều cao và rộng của window)
            # Giữ lại trục 2 (channels)
            s2_output[r, c, :] = np.max(window, axis=(0, 1))
            # Log cửa sổ và output tương ứng
            if r == 3:
                print(f"Max Pooling Window at (Row: {r_start}-{r_start+1}, Col: {c_start}-{c_start+1}):")
                print(window)
                print(f"Output at (Row: {r}, Col: {c}): {s2_output[r, c, :]}")

    # 3. Xuất ra file Golden
    print(f"Saving Golden Data to {GOLDEN_FILE}...")
    with open(GOLDEN_FILE, 'w') as f:
        # Flatten để ghi theo dòng giống format log của Verilog
        # Shape (14, 14, 6) -> (196, 6)
        flat_out = s2_output.reshape(-1, CHANNELS)
        for row in flat_out:
            line = " ".join(map(str, row))
            f.write(line + "\n")
            
    print("Done!")

if __name__ == "__main__":
    max_pooling_software()