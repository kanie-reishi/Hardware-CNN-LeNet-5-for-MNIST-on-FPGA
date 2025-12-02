# Hardware Accelerator for LeNet-5 CNN on FPGA

![Project Status](https://img.shields.io/badge/status-active-green)
![Language](https://img.shields.io/badge/verilog-systemverilog-blue)
![Python](https://img.shields.io/badge/python-3.x-yellow)

## 📖 Giới thiệu (Introduction)
Dự án này là một thiết kế phần cứng (Hardware Accelerator) cho mạng nơ-ron tích chập **LeNet-5** nhằm nhận diện chữ số viết tay (bộ dữ liệu MNIST). Thiết kế được viết bằng **Verilog HDL**, tối ưu hóa để triển khai trên FPGA.

Mục tiêu của dự án là xây dựng một hệ thống suy luận (Inference Engine) hiệu quả về tài nguyên, thực hiện các phép tính tích chập (Convolution) và lấy mẫu (Pooling) trực tiếp trên phần cứng.

## 📂 Cấu trúc dự án (Project Structure)
- **`model/`**: Chứa Golden Model viết bằng Python (PyTorch/TensorFlow). Dùng để huấn luyện mạng, lượng tử hóa (quantization) và xuất weights/biases dưới dạng file `.hex` hoặc `.txt`.
- **`rtl/`**: Source code Verilog cho các module phần cứng (Line Buffer, Window Array, Convolution Unit, v.v.).
- **`tb/`**: Các file Testbench dùng để mô phỏng và kiểm tra chức năng (Functional Verification).
- **`docs/`**: Tài liệu thiết kế, sơ đồ kiến trúc và kết quả mô phỏng.

## 🛠️ Công cụ sử dụng (Tools & Technologies)
- **Thiết kế & Mô phỏng:** Vivado / ModelSim / Quartus (Điền tool bạn dùng vào đây)
- **Ngôn ngữ:** Verilog HDL (RTL), Python (Model Reference).
- **Board FPGA:** (Ví dụ: Xilinx Zybo Z7-10, Altera DE10-Nano - Điền tên board của bạn)

## 🚀 Tính năng chính (Key Features)
- [x] **Line Buffer & Window Array:** Cơ chế trượt cửa sổ hiệu quả để xử lý dữ liệu ảnh streaming.
- [ ] **Convolution Layer:** Tính toán song song các kernel.
- [ ] **Quantization:** Chuyển đổi Floating-point sang Fixed-point (8-bit/16-bit) để tiết kiệm tài nguyên FPGA.
- [ ] **UART/VGA Interface:** Giao tiếp hiển thị kết quả (Dự kiến).

## ⚙️ Hướng dẫn chạy (How to Run)

### 1. Tạo Golden Data
Chạy script Python để train model và xuất file trọng số:
```bash
cd model
python train_lenet.pys
python export_weights.py

2. Mô phỏng phần cứngs
Sử dụng file testbench trong thư mục tb/ để chạy mô phỏng với file data vừa tạo: Load các file trong rtl/ và tb/ vào Vivado/ModelSim và chạy tb_top_module.

Kết quả(Results) sẽ được so sánh với Golden Model để xác nhận tính đúng đắn.

👨‍💻 Tác giả (Author)
- Tên: Hồ Chí Công
- Email: hcc82cva123@gmail.com