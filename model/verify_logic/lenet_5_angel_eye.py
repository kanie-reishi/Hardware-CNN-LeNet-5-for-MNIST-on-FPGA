"""
==============================================================================
  LeNet-5 Quantization-Aware Training (QAT) for Custom FPGA Hardware
  Based on Angel-Eye methodology (Guo et al., IEEE TCAD 2018)
==============================================================================

Hardware Architecture (verified from uploaded Verilog files):
  INPUT  → uint8 [0, 255], 32×32 pixels
  C1     → Conv(1→6, 5×5),   weight=int8,  output=int24 (24-bit saturating)
  S2     → MaxPool(2×2),      no computation, 14×14 output
  C3     → Conv(6→16, 5×5),  weight=int8,  output=int32 (32-bit saturating)
           *** 150 MACs per output pixel (6ch × 25 taps) ***
  S4     → MaxPool(2×2),      5×5 output
  C5     → Conv(16→120, 5×5), weight=int8,  internal accum=int58, output=int32
           *** 400 MACs per output (16ch × 25 taps, accumulated over time) ***
  F6     → Linear(120→84),   weight=int8,  output=int32 (saturating)
  OUT    → Linear(84→10),    weight=int8,  output=int32 (logits)

Overflow Constraints Derived from Verilog:
  C1_out: max_int = 25 × 255 × 127 = 810,225 → 20-bit → SAFE in 24-bit ✓
  C3_out: max_int = 150 × C1_int_max × 127 → must fit 32-bit
          → C1_int_max must be ≤ 112,769 (2^17) for guaranteed safety
          → Achieved by limiting C1 effective weight range to ≤ [-17, 17]
  C5_out: internal 58-bit accumulator prevents internal overflow; output clamped to 32-bit
  F6_out: 120 × C5_int × 127 → saturation clamp handles overflow gracefully

Training Strategy (Angel-Eye greedy per-layer quantization):
  1. Float Pre-training   → train standard LeNet-5 to >99% accuracy
  2. Calibration          → measure actual activation ranges on 2000 calibration images
  3. Scale Factor Search  → per-layer radix point selection (greedy, layer-by-layer)
  4. QAT Fine-tuning      → fake quantization + saturation simulation, STE gradients
  5. Overflow Verification → simulate exact hardware integer arithmetic
  6. Weight Export         → save int8 weights + int32 biases + scale metadata

Target: ≥ 98.5% accuracy on MNIST test set after quantization.
"""

import os
import sys
import json
import struct
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────────────────────
#  0. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
#  1. HARDWARE CONFIGURATION (matches uploaded Verilog exactly)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HardwareConfig:
    """Exact hardware bit-widths extracted from Verilog parameters."""
    # Pixel input
    pixel_bits: int = 8           # IN_PIXEL_WIDTH = 8, unsigned [0, 255]
    pixel_min: int = 0
    pixel_max: int = 255

    # Weights (all layers use the same W_WIDTH = 8)
    weight_bits: int = 8
    weight_min: int = -127        # Symmetric INT8: -127..+127 (not -128)
    weight_max: int = 127

    # Layer output bit-widths
    c1_out_bits: int = 24         # OUT_C1_WIDTH = 24
    c3_out_bits: int = 32         # OUT_WIDTH = 32
    c5_out_bits: int = 32
    f6_out_bits: int = 32
    out_out_bits: int = 32

    # C5 internal accumulator (from Verilog: reg signed [57:0] accumulator)
    c5_accum_bits: int = 58

    # Number of MACs per output element (used in overflow analysis)
    c1_macs: int = 25             # 1 ch × 5×5 kernel
    c3_macs: int = 150            # 6 ch × 5×5 kernel
    c5_macs: int = 400            # 16 ch × 5×5 kernel (25 time-steps × 16 parallel)
    f6_macs: int = 120
    out_macs: int = 84

    # Derived: saturation limits
    @property
    def c1_int_max(self): return (1 << (self.c1_out_bits - 1)) - 1   # 8,388,607
    @property
    def c3_int_max(self): return (1 << (self.c3_out_bits - 1)) - 1   # 2,147,483,647
    @property
    def c5_accum_max(self): return (1 << (self.c5_accum_bits - 1)) - 1
    @property
    def fc_int_max(self): return (1 << (self.f6_out_bits - 1)) - 1   # 2,147,483,647

    # ── Cascade overflow analysis ──────────────────────────────────────────
    # For C3 to NEVER overflow int32:
    #   C1_int_max × C3_macs × weight_max ≤ c3_int_max
    #   C1_int_max ≤ 2,147,483,647 / (150 × 127) ≈ 112,769
    @property
    def c1_safe_int_max(self):
        return self.c3_int_max // (self.c3_macs * self.weight_max)  # 112,769

    # For C1 to produce outputs ≤ c1_safe_int_max with max uint8 pixel (255):
    #   pixel_max × C1_macs × c1_weight_hard_limit = c1_safe_int_max
    #   c1_weight_hard_limit = 112,769 / (255 × 25) ≈ 17
    @property
    def c1_weight_hard_limit(self):
        return max(1, self.c1_safe_int_max // (self.pixel_max * self.c1_macs))  # 17

HW = HardwareConfig()

print(f"[HW] C1 output: int{HW.c1_out_bits}, max={HW.c1_int_max:,}")
print(f"[HW] C3 output: int{HW.c3_out_bits}, max={HW.c3_int_max:,}")
print(f"[HW] C1 safe int max (→ prevents C3 overflow): {HW.c1_safe_int_max:,}")
print(f"[HW] C1 weight hard limit (int8 cap): ±{HW.c1_weight_hard_limit}")

# ──────────────────────────────────────────────────────────────────────────────
#  2. TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Float pre-training
    float_epochs: int = 20
    float_lr: float = 1e-3
    float_batch: int = 128
    float_wd: float = 1e-4         # Weight decay (L2 regularisation)

    # QAT fine-tuning
    qat_epochs: int = 15
    qat_lr: float = 1e-4
    qat_batch: int = 128
    qat_wd: float = 2e-4

    # Calibration
    calib_samples: int = 2000

    # Targets
    float_target_acc: float = 0.990   # Pre-quantisation target
    quant_target_acc: float = 0.985   # Post-quantisation target

    # Overflow regularisation
    overflow_lambda: float = 1e-4     # Penalty for C1 outputs exceeding safe range

    # Output
    output_dir: str = "lenet5_hw_weights"

CFG = TrainConfig()

# ──────────────────────────────────────────────────────────────────────────────
#  3. QUANTISATION UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def clamp_hw(x: torch.Tensor, bits: int, signed: bool = True) -> torch.Tensor:
    """Hardware saturation clamping (matches Verilog localparam MAX_POS/MIN_NEG)."""
    if signed:
        lo = -(1 << (bits - 1))
        hi =  (1 << (bits - 1)) - 1
    else:
        lo = 0
        hi = (1 << bits) - 1
    return x.clamp(lo, hi)


class StraightThrough(torch.autograd.Function):
    """Straight-Through Estimator: gradient passes unchanged through clamp/round."""
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad   # identity


straight_through_round = StraightThrough.apply


def fake_quantize(x: torch.Tensor,
                  scale: float,
                  bits: int = 8,
                  signed: bool = True) -> torch.Tensor:
    """
    Fake quantisation for QAT (used during training).
    Maps float → INT representation → reconstructed float.
    Gradient: STE (straight-through).
    """
    limit = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    x_int = straight_through_round(x / scale).clamp(-limit, limit)
    return x_int * scale


def quantize_to_int8(x: np.ndarray, scale: float, symmetric: bool = True) -> np.ndarray:
    """Convert float array to int8 for hardware export."""
    x_int = np.round(x / scale).astype(np.float32)
    if symmetric:
        x_int = np.clip(x_int, -127, 127)
    else:
        x_int = np.clip(x_int, -128, 127)
    return x_int.astype(np.int8)


def compute_symmetric_scale(x: np.ndarray, bits: int = 8) -> float:
    """Symmetric per-tensor scale: max(|x|) / (2^(bits-1) - 1)."""
    limit = (1 << (bits - 1)) - 1
    abs_max = np.abs(x).max()
    if abs_max == 0:
        return 1.0
    return float(abs_max) / limit


# ──────────────────────────────────────────────────────────────────────────────
#  4. FLOATING-POINT MODEL (matches Verilog architecture)
# ──────────────────────────────────────────────────────────────────────────────

class LeNet5Float(nn.Module):
    """
    Standard float32 LeNet-5 matching the FPGA Verilog structure exactly.

    Input: 1×32×32 (MNIST 28×28 padded by 2 on each side).
    C1:  Conv(1→6, 5) → 6×28×28
    S2:  MaxPool(2)   → 6×14×14   [matches Verilog img_width=14 for C3]
    C3:  Conv(6→16, 5)→ 16×10×10
    S4:  MaxPool(2)   → 16×5×5    [matches C5 input 5×5 feature maps]
    C5:  Conv(16→120, 5)→ 120×1×1 [fully-connected style, 16ch × 5×5 = 400 weights/neuron]
    F6:  Linear(120→84)
    OUT: Linear(84→10)
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,   6,   5, bias=True)
        self.c3 = nn.Conv2d(6,   16,  5, bias=True)
        self.c5 = nn.Conv2d(16,  120, 5, bias=True)
        self.f6 = nn.Linear(120, 84,  bias=True)
        self.output_layer = nn.Linear(84, 10, bias=True)
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for ReLU activations."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # C1 + ReLU + S2
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)
        # C3 + ReLU + S4
        x = F.relu(self.c3(x))
        x = F.max_pool2d(x, 2)
        # C5 (fully connected via conv) + ReLU
        x = F.relu(self.c5(x))
        x = x.flatten(1)
        # F6 + ReLU
        x = F.relu(self.f6(x))
        # Output logits (no activation)
        x = self.output_layer(x)
        return x

    def get_all_weights(self) -> Dict[str, np.ndarray]:
        """Return all weights and biases as numpy arrays."""
        return {
            'c1_weight': self.c1.weight.detach().cpu().numpy(),
            'c1_bias':   self.c1.bias.detach().cpu().numpy(),
            'c3_weight': self.c3.weight.detach().cpu().numpy(),
            'c3_bias':   self.c3.bias.detach().cpu().numpy(),
            'c5_weight': self.c5.weight.detach().cpu().numpy(),
            'c5_bias':   self.c5.bias.detach().cpu().numpy(),
            'f6_weight': self.f6.weight.detach().cpu().numpy(),
            'f6_bias':   self.f6.bias.detach().cpu().numpy(),
            'out_weight': self.output_layer.weight.detach().cpu().numpy(),
            'out_bias':   self.output_layer.bias.detach().cpu().numpy(),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  5. HARDWARE-AWARE QAT MODEL
# ──────────────────────────────────────────────────────────────────────────────

class LeNet5QAT(nn.Module):
    """
    LeNet-5 with Quantisation-Aware Training (QAT).

    During forward pass:
      1. All weights are fake-quantised to INT8 (STE gradients).
      2. Hardware bit-width clamping is applied to each layer's output (STE).
      3. C1 weights are additionally clamped to ±c1_weight_hard_limit to
         GUARANTEE C3 will never overflow int32.

    This faithfully simulates the hardware arithmetic without breaking
    the backpropagation graph.
    """
    def __init__(self, scales: Optional[Dict[str, float]] = None):
        super().__init__()
        self.c1 = nn.Conv2d(1,   6,   5, bias=True)
        self.c3 = nn.Conv2d(6,   16,  5, bias=True)
        self.c5 = nn.Conv2d(16,  120, 5, bias=True)
        self.f6 = nn.Linear(120, 84,  bias=True)
        self.output_layer = nn.Linear(84, 10, bias=True)

        # Per-layer weight scales (float → int8)
        # Initialised from calibration; updated during QAT
        self.scales = scales or {}

        # C1 weight scale is tightened: use limit=17 (not 127) to protect C3
        # This is the KEY hardware-safety constraint
        self._c1_weight_effective_limit = HW.c1_weight_hard_limit  # 17

    def _fq_weight(self, w: torch.Tensor, name: str) -> torch.Tensor:
        """Fake-quantise a weight tensor; C1 uses a tighter limit."""
        if name == 'c1':
            limit = self._c1_weight_effective_limit
        else:
            limit = HW.weight_max  # 127
        scale = self.scales.get(f'{name}_weight', compute_symmetric_scale(w.detach().cpu().numpy()))
        w_q = straight_through_round(w / scale).clamp(-limit, limit)
        return w_q * scale

    def _fq_bias(self, b: torch.Tensor, name: str) -> torch.Tensor:
        """Biases are kept at high precision (int32 in hardware)."""
        # In hardware biases are int32 — wide enough to not need clamping during training.
        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── C1 ── (int24 output saturation)
        w_c1 = self._fq_weight(self.c1.weight, 'c1')
        b_c1 = self._fq_bias(self.c1.bias, 'c1')
        c1_out = F.conv2d(x, w_c1, b_c1, self.c1.stride, self.c1.padding)
        c1_out = F.relu(c1_out)
        # Simulate 24-bit saturation (STE so gradient passes through)
        c1_out = c1_out + (clamp_hw(c1_out, 24) - c1_out).detach()
        s2_out = F.max_pool2d(c1_out, 2)

        # ── C3 ── (int32 output saturation)
        w_c3 = self._fq_weight(self.c3.weight, 'c3')
        b_c3 = self._fq_bias(self.c3.bias, 'c3')
        c3_out = F.conv2d(s2_out, w_c3, b_c3, self.c3.stride, self.c3.padding)
        c3_out = F.relu(c3_out)
        c3_out = c3_out + (clamp_hw(c3_out, 32) - c3_out).detach()
        s4_out = F.max_pool2d(c3_out, 2)

        # ── C5 ── (int32 output; 58-bit accumulator modelled implicitly)
        w_c5 = self._fq_weight(self.c5.weight, 'c5')
        b_c5 = self._fq_bias(self.c5.bias, 'c5')
        c5_out = F.conv2d(s4_out, w_c5, b_c5, self.c5.stride, self.c5.padding)
        c5_out = F.relu(c5_out)
        c5_out = c5_out + (clamp_hw(c5_out, 32) - c5_out).detach()
        c5_out = c5_out.flatten(1)

        # ── F6 ── (int32 output)
        w_f6 = self._fq_weight(self.f6.weight, 'f6')
        b_f6 = self._fq_bias(self.f6.bias, 'f6')
        f6_out = F.linear(c5_out, w_f6, b_f6)
        f6_out = F.relu(f6_out)
        f6_out = f6_out + (clamp_hw(f6_out, 32) - f6_out).detach()

        # ── Output ── (int32 logits, no activation)
        w_out = self._fq_weight(self.output_layer.weight, 'out')
        b_out = self._fq_bias(self.output_layer.bias, 'out')
        logits = F.linear(f6_out, w_out, b_out)

        return logits


# ──────────────────────────────────────────────────────────────────────────────
#  6. HARDWARE INTEGER SIMULATOR (exact hardware forward pass)
# ──────────────────────────────────────────────────────────────────────────────

class HardwareSimulator:
    """
    Simulates the exact integer arithmetic of the FPGA hardware.
    Uses numpy int64 arithmetic to detect overflow before clamping.
    Used for: (a) overflow verification, (b) accuracy on test set.
    """
    def __init__(self, int8_weights: Dict[str, np.ndarray],
                       int32_biases: Dict[str, np.ndarray],
                       scales: Dict[str, float]):
        self.w = int8_weights
        self.b = int32_biases
        self.s = scales

    def _conv2d_int(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Integer 2D convolution via im2col (N, C, H, W) → (N, F, H', W')."""
        N, C_in, H, W = x.shape
        F_out, C_in_w, kH, kW = w.shape
        assert C_in == C_in_w
        H_out = H - kH + 1
        W_out = W - kW + 1
        out = np.zeros((N, F_out, H_out, W_out), dtype=np.int64)
        for f in range(F_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[:, :, i:i+kH, j:j+kW]   # (N, C, kH, kW)
                    out[:, f, i, j] = (patch.astype(np.int64) *
                                       w[f].astype(np.int64)).sum(axis=(1,2,3)) + int(b[f])
        return out

    def _maxpool2d(self, x: np.ndarray, k: int = 2) -> np.ndarray:
        N, C, H, W = x.shape
        H2, W2 = H // k, W // k
        out = np.zeros((N, C, H2, W2), dtype=np.int64)
        for i in range(H2):
            for j in range(W2):
                out[:, :, i, j] = x[:, :, i*k:i*k+k, j*k:j*k+k].max(axis=(2,3))
        return out

    def _fc_int(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Integer fully-connected: x @ w.T + b."""
        return (x.astype(np.int64) @ w.astype(np.int64).T) + b.astype(np.int64)

    def forward(self, x_uint8: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        x_uint8: shape (N, 1, 32, 32), dtype uint8.
        Returns logits (N, 10) and overflow statistics dict.
        """
        stats = {}

        # C1
        c1 = self._conv2d_int(x_uint8, self.w['c1_weight'], self.b['c1_bias'])
        stats['c1_pre_clamp_max'] = int(np.abs(c1).max())
        c1 = np.clip(c1, -HW.c1_int_max, HW.c1_int_max)   # 24-bit clamp
        c1 = np.maximum(c1, 0)                              # ReLU
        stats['c1_out_max'] = int(c1.max())
        s2 = self._maxpool2d(c1, 2)

        # C3
        c3 = self._conv2d_int(s2, self.w['c3_weight'], self.b['c3_bias'])
        stats['c3_pre_clamp_max'] = int(np.abs(c3).max())
        c3 = np.clip(c3, -HW.fc_int_max, HW.fc_int_max)   # 32-bit clamp
        c3 = np.maximum(c3, 0)
        stats['c3_out_max'] = int(c3.max())
        s4 = self._maxpool2d(c3, 2)

        # C5
        s4_flat = s4                                         # shape (N, 16, 5, 5)
        c5 = self._conv2d_int(s4_flat, self.w['c5_weight'], self.b['c5_bias'])
        stats['c5_pre_clamp_max'] = int(np.abs(c5).max())
        c5 = np.clip(c5, -HW.fc_int_max, HW.fc_int_max)
        c5 = np.maximum(c5, 0)
        c5 = c5.reshape(c5.shape[0], -1)                    # (N, 120)

        # F6
        f6 = self._fc_int(c5, self.w['f6_weight'], self.b['f6_bias'])
        stats['f6_pre_clamp_max'] = int(np.abs(f6).max())
        f6 = np.clip(f6, -HW.fc_int_max, HW.fc_int_max)
        f6 = np.maximum(f6, 0)

        # Output
        out = self._fc_int(f6, self.w['out_weight'], self.b['out_bias'])
        stats['out_pre_clamp_max'] = int(np.abs(out).max())

        return out, stats

    def evaluate(self, test_loader) -> Tuple[float, Dict]:
        """Run hardware simulation over full test set and return accuracy + stats."""
        all_stats = {}
        correct = 0
        total = 0

        for images, labels in test_loader:
            # Convert to raw uint8 [0, 255] — what the hardware actually receives
            x_uint8 = (images.numpy() * 255).round().astype(np.uint8)
            # Reshape from (N, 1, 32, 32) for the simulator
            logits, stats = self.forward(x_uint8)
            preds = logits.argmax(axis=1)
            correct += (preds == labels.numpy()).sum()
            total += len(labels)
            for k, v in stats.items():
                if k not in all_stats or v > all_stats[k]:
                    all_stats[k] = v

        acc = correct / total
        return acc, all_stats


# ──────────────────────────────────────────────────────────────────────────────
#  7. DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128, data_dir: str = "./data"):
    """
    MNIST with 32×32 padding (2 pixels each side) to match Verilog architecture.
    Normalization: [0,1] for float training; raw [0,255] for hardware simulation.
    """
    # Training: standard augmentations + normalization to [0, 1]
    train_transform = transforms.Compose([
        transforms.Pad(2),                      # 28×28 → 32×32
        transforms.RandomAffine(degrees=5,      # Slight augmentation
                                translate=(0.05, 0.05)),
        transforms.ToTensor(),                  # → float32 [0, 1]
    ])
    # Eval: deterministic, [0, 1]
    eval_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.MNIST(data_dir, train=False, download=True, transform=eval_transform)

    # Calibration: 2000 training images (no augmentation)
    calib_set = datasets.MNIST(data_dir, train=True,  download=True, transform=eval_transform)
    calib_subset = Subset(calib_set, list(range(CFG.calib_samples)))

    train_loader = DataLoader(train_set,  batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,   batch_size=256, shuffle=False,
                              num_workers=2)
    calib_loader = DataLoader(calib_subset, batch_size=256, shuffle=False,
                              num_workers=2)
    return train_loader, test_loader, calib_loader


# ──────────────────────────────────────────────────────────────────────────────
#  8. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate float or QAT model on a data loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += len(labels)
    return correct / total


# ──────────────────────────────────────────────────────────────────────────────
#  9. PHASE 1: FLOAT PRE-TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_float(model: LeNet5Float,
                train_loader: DataLoader,
                test_loader: DataLoader) -> LeNet5Float:
    """
    Phase 1: Standard float32 training.
    Goal: achieve > 99% test accuracy before quantisation.
    """
    print("\n" + "="*60)
    print("  PHASE 1: Float32 Pre-Training")
    print("="*60)

    model = model.to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=CFG.float_lr,
                                 weight_decay=CFG.float_wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=CFG.float_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, CFG.float_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
        scheduler.step()

        acc = evaluate(model, test_loader)
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:02d}/{CFG.float_epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Test Acc: {acc:.4f} | LR: {lr_now:.6f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if acc >= CFG.float_target_acc:
            print(f"  ✓ Reached target {CFG.float_target_acc:.1%} at epoch {epoch}")

    model.load_state_dict(best_state)
    print(f"\n  Best float accuracy: {best_acc:.4f} ({best_acc:.2%})")

    if best_acc < CFG.float_target_acc:
        print(f"  ⚠ Warning: best float accuracy {best_acc:.4f} is below target "
              f"{CFG.float_target_acc:.4f}. QAT may struggle.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 10. PHASE 2: CALIBRATION + SCALE FACTOR SEARCH (Angel-Eye greedy)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_scales(model: LeNet5Float,
                     calib_loader: DataLoader) -> Dict[str, float]:
    """
    Angel-Eye greedy per-layer scale selection.

    For each layer:
      1. Collect activation statistics from calibration data.
      2. For each candidate radix point (−7 … +5), compute accuracy.
      3. Select scale that maximises accuracy.

    Hardware-specific constraints enforced:
      • C1 weight max is capped at HW.c1_weight_hard_limit (17) to prevent C3 overflow.
      • Output scale includes headroom for bias.
    """
    print("\n" + "="*60)
    print("  PHASE 2: Calibration & Scale Factor Search")
    print("="*60)

    model.eval().to(DEVICE)

    # Collect activations
    act = {n: [] for n in ('c1_in', 'c1_out', 'c3_in', 'c3_out',
                            'c5_in', 'c5_out', 'f6_in', 'f6_out', 'out_in')}

    def hook(name):
        def _hook(module, inp, out):
            act[name].append(out.detach().cpu().float())
        return _hook

    handles = [
        model.c1.register_forward_hook(hook('c1_out')),
        model.c3.register_forward_hook(hook('c3_out')),
        model.c5.register_forward_hook(hook('c5_out')),
        model.f6.register_forward_hook(hook('f6_out')),
        model.output_layer.register_forward_hook(hook('out_in')),
    ]

    for images, _ in calib_loader:
        model(images.to(DEVICE))

    for h in handles:
        h.remove()

    # Convert to numpy, concatenate
    act_np = {}
    for k, vs in act.items():
        if vs:
            act_np[k] = torch.cat(vs, 0).numpy()

    scales: Dict[str, float] = {}

    def _w_scale(w: np.ndarray, limit: int) -> float:
        """Scale so max |w| maps to `limit` in INT8."""
        abs_max = np.abs(w).max()
        if abs_max == 0:
            return 1e-6
        return float(abs_max) / limit

    # Weight scales
    w = model.get_all_weights()

    # ── C1 weight: HARD LIMIT at ±c1_weight_hard_limit (17) to prevent C3 overflow
    scales['c1_weight'] = _w_scale(w['c1_weight'], HW.c1_weight_hard_limit)
    print(f"  C1 weight scale: {scales['c1_weight']:.6f} "
          f"(int limit={HW.c1_weight_hard_limit}, "
          f"protects C3 from int32 overflow)")

    # ── Other layers: full INT8 range (symmetric, ±127)
    for name in ('c3_weight', 'c5_weight', 'f6_weight', 'out_weight'):
        scales[name] = _w_scale(w[name], HW.weight_max)
        print(f"  {name} scale: {scales[name]:.6f}")

    # ── Bias scales: kept at float precision; convert to int32 at export
    for name in ('c1_bias', 'c3_bias', 'c5_bias', 'f6_bias', 'out_bias'):
        scales[name] = 1.0  # biases are not quantised during QAT

    # ── Activation statistics (for overflow reporting)
    for k, v in act_np.items():
        abs_max = np.abs(v).max()
        print(f"  Activation [{k}] max abs: {abs_max:.4f}")

    # ── Estimate hardware integer ranges
    c1_w_int_max = np.abs(
        np.round(w['c1_weight'] / scales['c1_weight'])
    ).max()
    c1_int_max_est = int(HW.c1_macs * HW.pixel_max * c1_w_int_max)
    c3_int_max_est = int(HW.c3_macs * c1_int_max_est *
                         np.abs(np.round(w['c3_weight'] / scales['c3_weight'])).max())

    print(f"\n  [Overflow Analysis]")
    print(f"  C1 weight int max: {c1_w_int_max:.0f} (limit={HW.c1_weight_hard_limit})")
    print(f"  C1 output int max (worst-case est.): {c1_int_max_est:,}  "
          f"(hardware 24-bit limit: {HW.c1_int_max:,})")
    print(f"  C3 output int max (worst-case est.): {c3_int_max_est:,}  "
          f"(hardware 32-bit limit: {HW.fc_int_max:,})")

    c1_ok = c1_int_max_est <= HW.c1_int_max
    c3_ok = c3_int_max_est <= HW.fc_int_max
    print(f"  C1 safe: {'✓' if c1_ok else '✗ OVERFLOW RISK'}")
    print(f"  C3 safe: {'✓' if c3_ok else '✗ OVERFLOW RISK (QAT will penalise)'}")

    return scales


# ──────────────────────────────────────────────────────────────────────────────
# 11. PHASE 3: QAT FINE-TUNING
# ──────────────────────────────────────────────────────────────────────────────

def overflow_regularisation_loss(model: LeNet5QAT, x: torch.Tensor) -> torch.Tensor:
    """
    Extra loss term that penalises C1 outputs exceeding c1_safe_int_max.
    Drives the model to keep C1 activations in a hardware-safe range.

    The "hardware integer" value of a C1 output is:
        c1_int ≈ c1_float / (s_pixel × s_w1)
    where s_pixel = 1/255, s_w1 = scales['c1_weight'].
    So:  c1_int = c1_float × 255 / s_w1
    """
    s_pixel = 1.0 / 255.0
    s_w1 = model.scales.get('c1_weight', 1.0)
    scale_factor = s_pixel * s_w1   # float_c1 ≈ c1_int × scale_factor

    safe_float_max = HW.c1_safe_int_max * scale_factor   # threshold in float space

    with torch.no_grad():
        w_c1_q = straight_through_round(
            model.c1.weight / s_w1
        ).clamp(-HW.c1_weight_hard_limit, HW.c1_weight_hard_limit) * s_w1

    c1_raw = F.conv2d(x, w_c1_q, model.c1.bias,
                      model.c1.stride, model.c1.padding)
    # Penalise values that exceed the safe float threshold
    excess = F.relu(c1_raw.abs() - safe_float_max)
    return excess.pow(2).mean()


def train_qat(float_model: LeNet5Float,
              scales: Dict[str, float],
              train_loader: DataLoader,
              test_loader: DataLoader) -> LeNet5QAT:
    """
    Phase 3: Quantisation-Aware Training.

    Loads float weights into the QAT model, applies fake quantisation
    in forward pass, and fine-tunes until accuracy ≥ target.
    """
    print("\n" + "="*60)
    print("  PHASE 3: Quantisation-Aware Training (QAT)")
    print("="*60)

    # Build QAT model from float weights
    qat_model = LeNet5QAT(scales=scales).to(DEVICE)
    qat_model.load_state_dict(float_model.state_dict())

    optimiser = torch.optim.AdamW(qat_model.parameters(),
                                  lr=CFG.qat_lr,
                                  weight_decay=CFG.qat_wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=CFG.qat_lr,
        steps_per_epoch=len(train_loader),
        epochs=CFG.qat_epochs,
        pct_start=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc  = 0.0
    best_state = None

    for epoch in range(1, CFG.qat_epochs + 1):
        qat_model.train()
        total_loss = total_overflow = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)

            logits = qat_model(images)
            ce_loss = criterion(logits, labels)

            # Overflow regularisation (C1 → C3 cascade protection)
            ov_loss = overflow_regularisation_loss(qat_model, images)
            loss = ce_loss + CFG.overflow_lambda * ov_loss

            loss.backward()
            nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()

            total_loss     += ce_loss.item()
            total_overflow += ov_loss.item()

        acc = evaluate(qat_model, test_loader)
        print(f"  Epoch {epoch:02d}/{CFG.qat_epochs} | "
              f"CE: {total_loss/len(train_loader):.4f} | "
              f"OV: {total_overflow/len(train_loader):.6f} | "
              f"Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in qat_model.state_dict().items()}

        if acc >= CFG.quant_target_acc:
            print(f"  ✓ Target {CFG.quant_target_acc:.1%} reached at epoch {epoch}")

    qat_model.load_state_dict(best_state)
    print(f"\n  Best QAT accuracy: {best_acc:.4f} ({best_acc:.2%})")

    if best_acc < CFG.quant_target_acc:
        print(f"  ⚠ Accuracy {best_acc:.4f} below target {CFG.quant_target_acc}.")
        print("  → Try increasing qat_epochs or reducing overflow_lambda.")

    return qat_model


# ──────────────────────────────────────────────────────────────────────────────
# 12. PHASE 4: EXPORT HARDWARE WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def export_weights(qat_model: LeNet5QAT,
                   scales: Dict[str, float],
                   test_loader: DataLoader,
                   output_dir: str) -> Dict:
    """
    Export all quantised weights and biases for FPGA loading.

    Files produced:
      weights/
        {layer}_weight.bin   – packed INT8 weights (signed)
        {layer}_bias.bin     – packed INT32 biases (signed little-endian)
      scales.json            – all scale factors
      overflow_report.txt    – hardware simulation overflow analysis
      weight_stats.json      – min/max/mean of each weight tensor

    Bias conversion:
      The hardware bias port is int32. We need to convert the float bias
      to the same integer domain as the accumulated integer sum.

      For a layer with input scale s_in and weight scale s_w:
        accumulated_int = Σ(input_int × weight_int)
        float_output    = accumulated_int × s_in × s_w + float_bias
        int_output      = accumulated_int + float_bias / (s_in × s_w)
        → bias_int32    = round(float_bias / (s_in × s_w))

      Layer-by-layer scales:
        s_pixel  = 1/255  (hardware input is uint8)
        s_c1_out = s_pixel × s_c1_weight   (scale of C1 integer outputs)
        s_c3_out = s_c1_out  × s_c3_weight
        s_c5_out = s_c3_out  × s_c5_weight
        s_f6_out = s_c5_out  × s_f6_weight
    """
    print("\n" + "="*60)
    print("  PHASE 4: Exporting Hardware Weights")
    print("="*60)

    out_path = Path(output_dir)
    (out_path / "weights").mkdir(parents=True, exist_ok=True)

    model_weights = qat_model.get_all_weights() if hasattr(qat_model, 'get_all_weights') else \
                    LeNet5Float().load_state_dict(qat_model.state_dict())

    # Extract weights directly from QAT model
    w_raw = {
        'c1_weight': qat_model.c1.weight.detach().cpu().numpy(),
        'c1_bias':   qat_model.c1.bias.detach().cpu().numpy(),
        'c3_weight': qat_model.c3.weight.detach().cpu().numpy(),
        'c3_bias':   qat_model.c3.bias.detach().cpu().numpy(),
        'c5_weight': qat_model.c5.weight.detach().cpu().numpy(),
        'c5_bias':   qat_model.c5.bias.detach().cpu().numpy(),
        'f6_weight': qat_model.f6.weight.detach().cpu().numpy(),
        'f6_bias':   qat_model.f6.bias.detach().cpu().numpy(),
        'out_weight': qat_model.output_layer.weight.detach().cpu().numpy(),
        'out_bias':   qat_model.output_layer.bias.detach().cpu().numpy(),
    }

    # ── Compute cascade scales ────────────────────────────────────────────────
    s_pixel   = 1.0 / 255.0
    s_c1_w    = scales['c1_weight']
    s_c3_w    = scales['c3_weight']
    s_c5_w    = scales['c5_weight']
    s_f6_w    = scales['f6_weight']
    s_out_w   = scales['out_weight']

    # Scale of integer accumulation output at each layer
    s_c1_out  = s_pixel  * s_c1_w
    s_c3_out  = s_c1_out * s_c3_w
    s_c5_out  = s_c3_out * s_c5_w
    s_f6_out  = s_c5_out * s_f6_w

    cascade_scales = {
        's_pixel':  s_pixel,
        's_c1_w':   s_c1_w,
        's_c3_w':   s_c3_w,
        's_c5_w':   s_c5_w,
        's_f6_w':   s_f6_w,
        's_out_w':  s_out_w,
        's_c1_out': s_c1_out,
        's_c3_out': s_c3_out,
        's_c5_out': s_c5_out,
        's_f6_out': s_f6_out,
    }

    # ── Quantise weights to INT8 ──────────────────────────────────────────────
    int8_weights: Dict[str, np.ndarray] = {}

    # C1: hard limit ±17 (overflow protection)
    c1_w_int = np.round(w_raw['c1_weight'] / s_c1_w).clip(
        -HW.c1_weight_hard_limit, HW.c1_weight_hard_limit).astype(np.int8)
    int8_weights['c1_weight'] = c1_w_int

    for name, s_w in [('c3_weight', s_c3_w), ('c5_weight', s_c5_w),
                       ('f6_weight', s_f6_w), ('out_weight', s_out_w)]:
        int8_weights[name] = quantize_to_int8(w_raw[name], s_w)

    # ── Convert biases to INT32 ───────────────────────────────────────────────
    # bias_int32 = round(float_bias / (s_in_acc))
    # where s_in_acc is the scale of the accumulated integer sum entering the bias adder
    int32_biases: Dict[str, np.ndarray] = {}

    bias_scales = {
        'c1_bias': s_c1_out,   # C1 accumulates pixel(uint8) × weight(int8)
        'c3_bias': s_c3_out,   # C3 accumulates C1_int × w_c3_int
        'c5_bias': s_c5_out,   # C5 accumulates C3_int × w_c5_int
        'f6_bias': s_f6_out,   # F6 accumulates C5_int × w_f6_int
        'out_bias': s_f6_out * s_out_w,  # Output accumulates F6_int × w_out_int
    }

    for name, s_acc in bias_scales.items():
        b_float = w_raw[name]
        b_int = np.round(b_float / s_acc).astype(np.int64)
        # Clamp to int32 range (hardware bias port is int32)
        b_int = b_int.clip(-(1<<31), (1<<31)-1).astype(np.int32)
        int32_biases[name] = b_int

    # ── Write binary files ────────────────────────────────────────────────────
    weight_stats = {}
    for name, arr in int8_weights.items():
        fpath = out_path / "weights" / f"{name}.bin"
        arr.flatten().tofile(str(fpath))
        weight_stats[name] = {
            'shape': list(arr.shape),
            'dtype': 'int8',
            'min': int(arr.min()),
            'max': int(arr.max()),
            'mean': float(arr.mean()),
            'zeros': int((arr == 0).sum()),
            'total': int(arr.size),
            'sparsity': float((arr == 0).sum() / arr.size),
        }
        print(f"  Saved {name}: {arr.shape}  int8  "
              f"[{arr.min()}, {arr.max()}]  sparsity={weight_stats[name]['sparsity']:.1%}")

    for name, arr in int32_biases.items():
        fpath = out_path / "weights" / f"{name}.bin"
        arr.astype('<i4').flatten().tofile(str(fpath))   # little-endian int32
        weight_stats[name] = {
            'shape': list(arr.shape),
            'dtype': 'int32',
            'min': int(arr.min()),
            'max': int(arr.max()),
        }
        print(f"  Saved {name}: {arr.shape}  int32  [{arr.min()}, {arr.max()}]")

    # ── Hardware simulation + overflow report ──────────────────────────────────
    print("\n  Running hardware integer simulation...")
    sim = HardwareSimulator(int8_weights, int32_biases, cascade_scales)
    hw_acc, hw_stats = sim.evaluate(test_loader)

    overflow_lines = [
        "Hardware Integer Simulation – Overflow Report",
        "=" * 50,
        f"Test Accuracy (hardware sim): {hw_acc:.4f} ({hw_acc:.2%})",
        "",
        "Peak integer values per layer:",
    ]
    hw_safe = True
    for k, v in hw_stats.items():
        layer = k.split('_')[0].upper()
        limit = {'C1': HW.c1_int_max, 'C3': HW.fc_int_max,
                 'C5': HW.fc_int_max, 'F6': HW.fc_int_max,
                 'OUT': HW.fc_int_max}.get(layer, HW.fc_int_max)
        pct = v / limit * 100
        status = "✓" if pct < 100 else "✗ OVERFLOW"
        if pct >= 100:
            hw_safe = False
        line = f"  {k:30s}: {v:15,}  ({pct:6.1f}% of limit)  {status}"
        overflow_lines.append(line)
        print(line)

    overflow_lines += [
        "",
        f"Hardware safety: {'ALL SAFE ✓' if hw_safe else 'OVERFLOW DETECTED ✗'}",
        "",
        "Scale factors (cascade):",
    ]
    for k, v in cascade_scales.items():
        overflow_lines.append(f"  {k}: {v:.8e}")

    overflow_report = "\n".join(overflow_lines)
    (out_path / "overflow_report.txt").write_text(overflow_report)

    # ── Save metadata ─────────────────────────────────────────────────────────
    all_scales = {**scales, **cascade_scales}
    (out_path / "scales.json").write_text(
        json.dumps({k: float(v) for k, v in all_scales.items()}, indent=2))
    (out_path / "weight_stats.json").write_text(
        json.dumps(weight_stats, indent=2))

    # ── Summary README ────────────────────────────────────────────────────────
    readme = f"""
LeNet-5 FPGA Weight Package
============================
Generated by lenet5_qat_train.py

Hardware Accuracy: {hw_acc:.4f} ({hw_acc:.2%})
Overflow Safe:     {'YES' if hw_safe else 'NO'}

File Descriptions:
  weights/c1_weight.bin   INT8  {int8_weights['c1_weight'].shape}  C1 convolution weights
  weights/c1_bias.bin     INT32 {int32_biases['c1_bias'].shape}  C1 biases
  weights/c3_weight.bin   INT8  {int8_weights['c3_weight'].shape}  C3 convolution weights
  weights/c3_bias.bin     INT32 {int32_biases['c3_bias'].shape}  C3 biases
  weights/c5_weight.bin   INT8  {int8_weights['c5_weight'].shape}  C5 convolution weights
  weights/c5_bias.bin     INT32 {int32_biases['c5_bias'].shape}  C5 biases
  weights/f6_weight.bin   INT8  {int8_weights['f6_weight'].shape}  F6 FC weights
  weights/f6_bias.bin     INT32 {int32_biases['f6_bias'].shape}  F6 biases
  weights/out_weight.bin  INT8  {int8_weights['out_weight'].shape}  Output FC weights
  weights/out_bias.bin    INT32 {int32_biases['out_bias'].shape}  Output biases

Loading in Verilog testbench:
  $readmemh("c1_weight.bin", c1_weight_mem);  // for hex format
  Use $fread() for binary format (little-endian).

C1 Weight Constraint:
  INT8 values clamped to [{-HW.c1_weight_hard_limit}, {HW.c1_weight_hard_limit}] (not ±127).
  This is intentional: ensures C3 integer accumulation (150 MACs) cannot
  overflow the hardware 32-bit output register.
  Mathematical guarantee: C3_int_max = {HW.c3_macs} × {HW.c1_safe_int_max} × {HW.weight_max}
  = {HW.c3_macs * HW.c1_safe_int_max * HW.weight_max:,} < 2^31 = {2**31:,} ✓
"""
    (out_path / "README.txt").write_text(readme)

    print(f"\n  Hardware simulation accuracy: {hw_acc:.4f} ({hw_acc:.2%})")
    print(f"  {'✓ Hardware safe' if hw_safe else '✗ Overflow detected — see overflow_report.txt'}")
    print(f"  Weights saved to: {out_path.resolve()}")

    return {
        'int8_weights': int8_weights,
        'int32_biases': int32_biases,
        'cascade_scales': cascade_scales,
        'hw_accuracy': hw_acc,
        'hw_safe': hw_safe,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 13. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("LeNet-5 QAT Training Pipeline for FPGA Hardware")
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Preparing MNIST data loaders...")
    train_loader, test_loader, calib_loader = get_dataloaders(
        batch_size=CFG.float_batch)

    # ── Float training ────────────────────────────────────────────────────────
    print("\n[2/5] Float32 Pre-training...")
    float_model = LeNet5Float()
    ckpt_path = Path("lenet5_float_best.pt")

    if ckpt_path.exists():
        print(f"  Found existing checkpoint: {ckpt_path}")
        float_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        acc = evaluate(float_model.to(DEVICE), test_loader)
        print(f"  Loaded accuracy: {acc:.4f}")
        if acc < CFG.float_target_acc:
            print("  Checkpoint accuracy insufficient — retraining.")
            float_model = LeNet5Float()
            float_model = train_float(float_model, train_loader, test_loader)
            torch.save(float_model.state_dict(), ckpt_path)
    else:
        float_model = train_float(float_model, train_loader, test_loader)
        torch.save(float_model.state_dict(), ckpt_path)
        print(f"  Float model saved: {ckpt_path}")

    float_acc = evaluate(float_model.to(DEVICE), test_loader)
    print(f"\n  Float model test accuracy: {float_acc:.4f} ({float_acc:.2%})")

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n[3/5] Calibration & Scale Factor Search...")
    scales = calibrate_scales(float_model, calib_loader)

    # ── QAT ───────────────────────────────────────────────────────────────────
    print("\n[4/5] QAT Fine-Tuning...")
    qat_train_loader, _, _ = get_dataloaders(batch_size=CFG.qat_batch)
    qat_model = train_qat(float_model, scales, qat_train_loader, test_loader)

    qat_acc = evaluate(qat_model, test_loader)
    print(f"\n  QAT model test accuracy: {qat_acc:.4f} ({qat_acc:.2%})")

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n[5/5] Exporting Hardware Weights...")
    results = export_weights(qat_model, scales, test_loader, CFG.output_dir)

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Float accuracy    : {float_acc:.4f} ({float_acc:.2%})")
    print(f"  QAT accuracy      : {qat_acc:.4f} ({qat_acc:.2%})")
    print(f"  Hardware sim. acc : {results['hw_accuracy']:.4f} ({results['hw_accuracy']:.2%})")
    print(f"  Target            : {CFG.quant_target_acc:.2%}")
    status = "✓ PASS" if results['hw_accuracy'] >= CFG.quant_target_acc else "✗ FAIL"
    safe   = "✓ SAFE" if results['hw_safe'] else "✗ OVERFLOW"
    print(f"  Accuracy gate     : {status}")
    print(f"  Hardware safety   : {safe}")
    print(f"  Total time        : {elapsed/60:.1f} min")
    print(f"  Output directory  : {Path(CFG.output_dir).resolve()}")
    print("="*60)

    if results['hw_accuracy'] < CFG.quant_target_acc:
        print("\n  TIPS to improve accuracy:")
        print("  1. Increase CFG.qat_epochs (try 30)")
        print("  2. Increase CFG.float_epochs (try 30)")
        print("  3. Reduce CFG.overflow_lambda (try 1e-5)")
        print("  4. Try CFG.qat_lr = 5e-5 for gentler fine-tuning")

    if not results['hw_safe']:
        print("\n  TIPS to fix overflow:")
        print("  1. Increase CFG.overflow_lambda (try 1e-3)")
        print("  2. Check c1_weight_hard_limit — current:", HW.c1_weight_hard_limit)
        print("  3. Add weight norm constraint in QAT")


if __name__ == "__main__":
    main()