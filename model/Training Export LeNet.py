import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import verify_hex
# ==========================================
# 1. ĐỊNH NGHĨA MODEL (Hardware Friendly)
# ==========================================
class LeNet5_Hardware(nn.Module):
    def __init__(self):
        super(LeNet5_Hardware, self).__init__()
        # Lưu ý: bias=False để đơn giản hóa phần cứng (bớt bộ cộng)
        # Padding=2 để ảnh 28x28 -> 32x32
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Input FC1: 16 kênh * 5 * 5 = 400
        self.c5 = nn.Linear(400, 120, bias=False)
        self.f6 = nn.Linear(120, 84, bias=False)
        self.output = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = self.s2(self.relu(self.c1(x)))
        x = self.s4(self.relu(self.c3(x)))
        x = self.flatten(x)
        x = self.relu(self.c5(x))
        x = self.relu(self.f6(x))
        x = self.output(x)
        return x

# ==========================================
# 2. HÀM XUẤT DATA CHO VERILOG (QUAN TRỌNG)
# ==========================================
def export_weight_to_hex(weight_tensor, filename):
    # Chuyển float -> fixed point (Ví dụ: Q8.8 hoặc giữ nguyên float nếu làm float)
    # Ở đây để đơn giản cho bước 1, ta xuất Float dạng HEX chuẩn IEEE 754
    # Hoặc xuất dạng số thực text để dễ debug trước.
    # Tốt nhất cho người mới: Nhân với 256 (<<8) rồi ép kiểu int (Fixed-point Q8.8)
    
    with open(filename, 'w') as f:
        w_flat = weight_tensor.detach().numpy().flatten()
        for w in w_flat:
            # Scale cho 8-bit (Max range -128..127)
            # Chọn Q1.6 (nhân 64) để an toàn cho các trọng số khoảng -2.0 đến 2.0
            w_fixed = int(w * 64) 
            
            # Kẹp giá trị (Clamp) để tránh tràn số
            w_fixed = max(min(w_fixed, 127), -128)
            
            # Xử lý số âm (bù 2 cho 8 bit)
            if w_fixed < 0:
                w_fixed = (1 << 8) + w_fixed
                
            # Giữ 8 bit
            w_fixed = w_fixed & 0xFF

            f.write(f"{w_fixed:02x}\n") # Ghi hex 2 ký tự
    print(f"Saved {filename} with {len(w_flat)} params.")

def export_image_to_hex(image_tensor, label, filename):
    """
    Xuất ảnh Tensor ra file Hex.
    Tự động Pad từ 28x28 lên 32x32 nếu cần thiết.
    """
    # 1. Kiểm tra kích thước và Padding
    # image_tensor shape thường là [1, 28, 28] hoặc [28, 28]
    if image_tensor.shape[-1] == 28:
        # Cú pháp pad: (Left, Right, Top, Bottom)
        # Pad thêm 2 pixel số 0 vào mỗi cạnh
        image_tensor = F.pad(image_tensor, (2, 2, 2, 2), "constant", 0)
    
    # Lúc này image_tensor đã là 32x32
    with open(filename, 'w') as f:
        img_flat = image_tensor.numpy().flatten()
        # Kiểm tra length img_flat phải là 1024
        assert len(img_flat) == 32*32, "Image size after padding must be 32x32!"
        for pixel in img_flat:
            # Pixel MNIST gốc là 0.0 -> 1.0 hoặc 0 -> 255
            # Nếu tensor đã chuẩn hóa, ta nhân lại về 255
            # Input đầu vào cho phần cứng thường là 8-bit unsigned
            p_val = int(pixel * 255) if pixel <= 1.0 else int(pixel)
            p_val = p_val & 0xFF
            f.write(f"{p_val:02x}\n")

# ==========================================
# 3. MAIN: TRAIN & EXPORT
# ==========================================
def main():
    # Setup
    device = torch.device("cpu") # Dùng CPU cho đơn giản
    os.makedirs("verilog_data", exist_ok=True)
    
    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Init Model
    model = LeNet5_Hardware().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train nhanh 1 epoch (đủ demo)
    print("Start Training...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss {loss.item():.4f}")
            if batch_idx == 500: break # Dừng sớm cho nhanh

    # Export Weights
    print("\nExporting Weights for Verilog...")
    export_weight_to_hex(model.c1.weight, "verilog_data/c1_weight.hex")
    export_weight_to_hex(model.c1.weight, "verify_c1/c1_weight.hex")
    #verify_hex.verify_weights(model.c1.weight, "verilog_data/c1_weight.hex")
    export_weight_to_hex(model.c3.weight, "verilog_data/c3_weight.hex")
    #verify_hex.verify_weights(model.c3.weight, "verilog_data/c3_weight.hex")
    export_weight_to_hex(model.c5.weight, "verilog_data/c5_weight.hex")
    #verify_hex.verify_weights(model.c5.weight, "verilog_data/c5_weight.hex")
    export_weight_to_hex(model.f6.weight, "verilog_data/f6_weight.hex")
    #verify_hex.verify_weights(model.f6.weight, "verilog_data/f6_weight.hex")
    export_weight_to_hex(model.output.weight, "verilog_data/output_weight.hex")
    #verify_hex.verify_weights(model.output.weight, "verilog_data/output_weight.hex")

    # Export 5 ảnh test
    print("\nExporting Test Images...")
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    for i in range(5):
        img, label = test_dataset[i]
        export_image_to_hex(img, label, f"verilog_data/test_img_{i}_label_{label}.hex")
        export_image_to_hex(img, label, f"verify_c1/test_img_{i}_label_{label}.hex")
        export_image_to_hex(img, label, f"verify_lb/test_img_{i}_label_{label}.hex")

    print("\nDONE! Check folder 'verilog_data'")

if __name__ == '__main__':
    main()