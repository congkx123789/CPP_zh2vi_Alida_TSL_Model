#!/usr/bin/env python3
"""
Export Dynamic Batch ONNX Model (INT8 Quantized) for C++ Native GPU Acceleration.
Enables dynamic batch size [Batch_Size, 64] on ONNX Runtime CUDA Execution Provider.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'TSL_Translator_Standalone'))

from models.student_nat import NonAutoregressiveStudentModel

class DynamicNATWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src):
        src_emb = self.model.pos_encoder(self.model.src_embedding(src) * math.sqrt(self.model.d_model))
        enc_out = self.model.encoder(src_emb)
        fertility_pred = self.model.fertility_module(enc_out).squeeze(-1)

        max_tgt_len = 64
        upsampled = F.interpolate(enc_out.transpose(1, 2), size=max_tgt_len, mode='linear', align_corners=False).transpose(1, 2)
        upsampled_emb = self.model.pos_encoder(upsampled)

        dec_out = self.model.decoder(upsampled_emb)
        logits = self.model.fc_out(dec_out)
        return logits, fertility_pred

def main():
    print("================================================================================")
    print("🚀 EXPORTING DYNAMIC BATCH INT8 ONNX MODEL FOR NATIVE C++ GPU INFERENCE")
    print("================================================================================")

    ckpt_path = os.path.join(SCRIPT_DIR, '..', 'Alida_TSL_Model', 'checkpoints', 'stage2_student_nat_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SCRIPT_DIR, '..', 'TSL_Translator_Standalone', 'checkpoints', 'stage2_student_nat_best.pt')

    model = NonAutoregressiveStudentModel(25015, 18004)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = DynamicNATWrapper(model)

    dummy_src = torch.randint(1, 25000, (2, 64), dtype=torch.long)
    onnx_fp32_path = os.path.join(SCRIPT_DIR, 'model', 'student_nat_fp32.onnx')
    onnx_int8_path = os.path.join(SCRIPT_DIR, 'model', 'student_nat_int8.onnx')

    os.makedirs(os.path.dirname(onnx_fp32_path), exist_ok=True)

    print(f"📦 Exporting Dynamic Batch FP32 ONNX model to: {onnx_fp32_path}...")
    torch.onnx.export(
        wrapper,
        dummy_src,
        onnx_fp32_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['src'],
        output_names=['logits', 'fertility_pred'],
        dynamic_axes={
            'src': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'fertility_pred': {0: 'batch_size'}
        }
    )

    print(f"⚡ Quantizing FP32 ONNX model to INT8: {onnx_int8_path}...")
    quantize_dynamic(
        onnx_fp32_path,
        onnx_int8_path,
        weight_type=QuantType.QUInt8
    )

    if os.path.exists(onnx_fp32_path):
        os.remove(onnx_fp32_path)

    sz_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"✅ Dynamic Batch INT8 ONNX Model ready! File size: {sz_mb:.2f} MB")
    print("================================================================================")

if __name__ == '__main__':
    main()
