import torch
import torch.nn as nn
import numpy as np

# --- CẤU HÌNH ---
# Giả lập input từ S4 (hoặc load từ file của bạn)
# S4 Output: (1 Batch, 16 Channels, 5 Height, 5 Width) flattened
INPUT_S4_FILE = "output_s4_log.txt" 

# Tên các file Hex sẽ tạo ra cho FPGA
HEX_C5 = "c5_weight.hex"
HEX_F6 = "f6_weight.hex"
HEX_OUT = "output_weight.hex"

# File Golden Output để check Waveform
GOLDEN_C5 = "golden_c5_ref.txt"
GOLDEN_F6 = "golden_f6_ref.txt"
GOLDEN_FINAL = "golden_final_ref.txt"

# --- MÔ HÌNH LENET5 GIẢ LẬP (ĐỂ LẤY TRỌNG SỐ) ---
# Bạn có thể thay phần này bằng code load state_dict từ file .pth của bạn
class LeNet5_Part(nn.Module):
    def __init__(self):
        super().__init__()
        # C5: Input 400 (16x5x5) -> Output 120
        self.c5 = nn.Linear(16 * 5 * 5, 120)
        # F6: Input 120 -> Output 84
        self.f6 = nn.Linear(120, 84)
        # Output: Input 84 -> Output 10
        self.out = nn.Linear(84, 10)

def save_hex(tensor_flat, filename):
    """Lưu tensor ra file hex cho Verilog $readmemh"""
    # Chuyển sang numpy và ép kiểu int 8-bit hoặc 32-bit tùy ý
    # Ở đây giả sử trọng số đã được Quantize sang int8
    # Nếu trọng số là float, bạn cần nhân với scale factor trước
    data = tensor_flat.detach().numpy().astype(int)
    
    with open(filename, 'w') as f:
        for val in data:
            # Xử lý số âm cho Hex (Two's complement 8-bit)
            if val < 0: val += 256 # Với 8-bit weight
            f.write(f"{val:02X}\n")
    print(f"✅ Đã lưu {filename} ({len(data)} weights)")

def main():
    # 1. Khởi tạo model và trọng số (Giả lập int8)
    model = LeNet5_Part()
    
    # Random trọng số integer nhỏ để dễ debug
    with torch.no_grad():
        model.c5.weight.data = torch.randint(-1, 2, (120, 400)).double() # Giảm xuống {-1, 0, 1} để an toàn tuyệt đối
        model.c5.bias.data.fill_(0)
        
        model.f6.weight.data = torch.randint(-1, 2, (84, 120)).double()
        model.f6.bias.data.fill_(0)
        
        model.out.weight.data = torch.randint(-1, 2, (10, 84)).double()
        model.out.bias.data.fill_(0)

    # ==========================================
    # PHẦN 1: XỬ LÝ WEIGHT C5 (QUAN TRỌNG NHẤT)
    # ==========================================
    print("\n--- XỬ LÝ C5 ---")
    # Shape gốc PyTorch: (120, 400) tương ứng (Out, In_16 * H_5 * W_5)
    # Thứ tự gốc: [Ch0_Px0, Ch0_Px1... Ch0_Px24, Ch1_Px0...] -> NCHW
    w_c5_orig = model.c5.weight.data.clone()
    
    # 1. Reshape về 3 chiều: (Out, In_Channel, Pixel)
    w_c5_3d = w_c5_orig.view(120, 16, 25)
    
    # 2. Transpose để khớp Hardware: (Out, Pixel, In_Channel)
    # Hardware đọc: Loop Pixel -> Loop 16 Channel song song
    w_c5_hw = w_c5_3d.permute(0, 2, 1).contiguous() # Shape: (120, 25, 16)
    
    # 3. Flatten và lưu Hex
    save_hex(w_c5_hw.view(-1), HEX_C5)

    # ==========================================
    # PHẦN 2: XỬ LÝ WEIGHT F6 & OUTPUT
    # ==========================================
    # F6 và Output thường là nhân ma trận Vector x Matrix.
    # Nếu HW của bạn tính F6 theo kiểu: Tính xong 1 neuron output rồi mới tính neuron tiếp theo (Sequential Output)
    # Thì thứ tự mặc định của PyTorch (Row-Major) là ĐÚNG.
    # PyTorch: [W_out0_in0, W_out0_in1...], [W_out1_in0...]
    
    print("\n--- XỬ LÝ F6 & OUTPUT ---")
    # F6: (84, 120) -> Lưu thẳng, không cần transpose nếu HW tính từng dòng
    save_hex(model.f6.weight.data.view(-1), HEX_F6)
    
    # Output: (10, 84) -> Lưu thẳng
    save_hex(model.out.weight.data.view(-1), HEX_OUT)

    # ==========================================
    # PHẦN 3: TÍNH GOLDEN DATA (SIMULATION)
    # ==========================================
    print("\n--- TÍNH TOÁN GOLDEN DATA ---")
    
    # 1. Load Input S4 giả lập (hoặc từ file)
    try:
        # Giả sử file log S4 lưu 16 số trên 1 dòng
        s4_data = np.loadtxt(INPUT_S4_FILE) # Shape (25, 16) nếu là 1 ảnh
        if len(s4_data.shape) == 1: s4_data = s4_data.reshape(25, 16)
        
        # Chuyển S4 data về dạng PyTorch Flatten (Channel First) để nhân với weight gốc
        # S4 Log Hardware: (Pixel 0..24, Ch 0..15)
        # Cần Transpose về: (Ch 0..15, Pixel 0..24) sau đó flatten
        input_tensor = torch.tensor(s4_data, dtype=torch.double) # (25, 16) Dùng double
        
        # Quan trọng: Input vào PyTorch Linear phải khớp thứ tự PyTorch Linear Weight
        # Input tensor hiện tại là (Pixel, Channel). 
        # Cần đổi thành (Channel, Pixel) rồi flatten
        input_pytorch = input_tensor.t().contiguous().view(-1) # (400,)
        
    except:
        print("⚠️ Không đọc được input S4, dùng random input.")
        input_pytorch = torch.randint(0, 5, (400,)).double()

    # 2. Tính C5 (Dùng weight gốc PyTorch vì input đã chuẩn hóa theo PyTorch)
    # Linear: y = xA^T + b. PyTorch thực hiện: output = input @ weight.t()
    c5_out = torch.matmul(input_pytorch, model.c5.weight.data.t()) + model.c5.bias.data
    
    # Saturation giả lập (cho giống HW)
    c5_out_clamped = torch.clamp(c5_out, -2147483648, 2147483647)
    np.savetxt(GOLDEN_C5, c5_out_clamped.numpy().astype(int), fmt='%d')
    print(f"   -> C5 Range: [{c5_out_clamped.min()}, {c5_out_clamped.max()}]")
    print(f"✅ Đã lưu {GOLDEN_C5}")

    # 3. Tính F6
    # F6 Input là Output của C5
    f6_out = torch.matmul(c5_out_clamped, model.f6.weight.data.t()) + model.f6.bias.data
    f6_out = np.maximum(0, f6_out) # <--- (ReLU)
    f6_out_clamped = torch.clamp(f6_out, -2147483648, 2147483647)
    np.savetxt(GOLDEN_F6, f6_out_clamped.numpy().astype(int), fmt='%d')
    print(f"   -> F6 Range: [{f6_out_clamped.min()}, {f6_out_clamped.max()}]")
    print(f"✅ Đã lưu {GOLDEN_F6}")

    # 4. Tính Final Output
    final_out = torch.matmul(f6_out_clamped, model.out.weight.data.t()) + model.out.bias.data
    final_out_clamped = torch.clamp(final_out, -2147483648, 2147483647)
    np.savetxt(GOLDEN_FINAL, final_out_clamped.numpy().astype(int), fmt='%d')
    print(f"   -> Final Range: [{final_out_clamped.min()}, {final_out_clamped.max()}]")
    
    # In ra kết quả dự đoán (Argmax)
    pred_idx = torch.argmax(final_out_clamped).item()
    print(f"   -> 🎯 MODEL DỰ ĐOÁN: SỐ {pred_idx} (Score: {final_out_clamped[pred_idx].item()})")
    print(f"✅ Đã lưu {GOLDEN_FINAL}")

if __name__ == "__main__":
    main()