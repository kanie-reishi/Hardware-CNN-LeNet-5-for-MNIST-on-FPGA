LeNet-5 FPGA Weight Package (Bit-Shift Architecture)
======================================================
HW sim accuracy : 0.9948 (99.48%)
Target          : 98.50%
Safety          : ALL OK

FILES
-----
weights/*.bin              Binary (little-endian)
weights/txt/*.txt          Human-readable decimal
weights/hex/*.hex          Hex for $readmemh
all_weights_inspection.txt All layers in one file
overflow_report.txt        Accumulator peaks + safety analysis
scales.json                All scale factors
right_shifts.json          Right-shift values for Verilog

RIGHT-SHIFT PARAMETERS (CRITICAL - program these into hardware):
  localparam RS_C1 = 12;  // C1: 809,625 >> 12 = 197/127
  localparam RS_C3 = 9;  // C3: MACs x INT8 x INT8
  localparam RS_C5 = 9;  // C5: MACs x INT8 x INT8
  localparam RS_F6 = 7;  // F6: MACs x INT8 x INT8
  // Output layer: NO shift (raw INT32 logits, compare for argmax)

VERILOG ROUND-SHIFT IMPLEMENTATION:
  // For any layer with rs > 0:
  // (Using C1 as example, RS_C1 = 12)
  wire [7:0] c1_shifted =
      (c1_relu_int + (1 << (RS_C1-1))) >> RS_C1;
  wire [7:0] c1_out = (c1_shifted > 127) ? 8'd127 : c1_shifted[7:0];
  // The (> 127) check is a safety net; it should never trigger.

WEIGHT ORDERING (PyTorch = C row-major):
  Conv: (out_ch, in_ch, kH, kW)  -- c1[6][1][5][5], c3[16][6][5][5]
  FC:   (out_neurons, in_neurons) -- f6[84][120], out[10][84]

BIAS CONVERSION (bias added BEFORE right-shift):
  bias_int32 = round(bias_float / s_acc_layer)
  C1  s_acc = 1.4474e-05  (= 1/255 x s_c1_w)
  C3  s_acc = 1.8243e-04  (= s_c1_out x s_c3_w)
  C5  s_acc = 2.6970e-04  (= s_c3_out x s_c5_w)
  F6  s_acc = 6.0251e-04  (= s_c5_out x s_f6_w)
  OUT s_acc = 7.5416e-04 (= s_f6_out x s_out_w)

ALL WEIGHTS USE FULL INT8 RANGE +-127:
  No artificial c1_weight_limit = 10 constraint needed.
  Cascade overflow is prevented by right-shifting C1 output to INT8
  BEFORE feeding it to C3. C3 receives INT8 (max 127), not 24-bit.
  Max C3 accumulation = 2,419,350 << 2,147,483,647 (INT32 max)
