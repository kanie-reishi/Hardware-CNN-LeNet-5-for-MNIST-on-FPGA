import numpy as np

# CẤU HÌNH
INPUT_FILE = "output_c3_log.txt"   # Input từ C3 (10x10, 16 kênh)
OUTPUT_FILE = "golden_s4_ref.txt"  # Output mong đợi S4 (5x5, 16 kênh)
IN_SIZE = 10
OUT_SIZE = 5
CHANNELS = 16

def main():
    print(f"Reading {INPUT_FILE}...")
    data = []
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                if not line.strip(): continue
                parts = [int(x) for x in line.strip().split()]
                if len(parts) == CHANNELS:
                    data.append(parts)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy {INPUT_FILE}. Chạy TB C3 trước!")
        return

    # Reshape: (100 dòng, 16 cột) -> (10, 10, 16)
    np_data = np.array(data)
    # Cắt cho đúng kích thước nếu file log bị dư
    if np_data.shape[0] > IN_SIZE * IN_SIZE:
        np_data = np_data[:IN_SIZE * IN_SIZE]
    
    feature_maps = np_data.reshape(IN_SIZE, IN_SIZE, CHANNELS)
    
    # Max Pooling Logic
    output_maps = np.zeros((OUT_SIZE, OUT_SIZE, CHANNELS), dtype=int)
    
    print("Computing Max Pooling S4...")
    for r in range(OUT_SIZE):
        for c in range(OUT_SIZE):
            # Window 2x2
            r_start = r * 2
            c_start = c * 2
            window = feature_maps[r_start:r_start+2, c_start:c_start+2, :]
            # Max pooling dọc trục 0 và 1 (h và w)
            output_maps[r, c, :] = np.max(window, axis=(0, 1))

    # Save to file
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        # Flatten (25 dòng, 16 cột)
        flat_out = output_maps.reshape(-1, CHANNELS)
        for row in flat_out:
            line = " ".join(map(str, row))
            f.write(line + "\n")
            
    print("✅ Done!")

if __name__ == "__main__":
    main()