from safetensors import safe_open
file_path = 'model-00001-of-00074.safetensors'

# 使用 safe_open 打开文件
with safe_open(file_path, framework="pt") as f:
    # 获取所有键
    # keys = f.keys()
    # print("Keys in the safetensors file:", keys)
    tensors = {}
    keys = [
        #     'model.layers.0.mlp.down_proj.input_scale', 
            'model.layers.0.mlp.down_proj.qweight', 
            'model.layers.0.mlp.down_proj.qzeros', 
            'model.layers.0.mlp.down_proj.scales']
    for key in keys:
        tensor = f.get_tensor(key)
        print(f"Tensor '{key}': shape={tensor.shape}, dtype={tensor.dtype}")

# import torch

# def unpack_uint8_to_fp4(x: torch.Tensor) -> torch.Tensor:
#     assert x.dtype == torch.uint8, "Input tensor must be of type torch.uint8"
    
#     # 解包低4位和高4位，并合并到新维度
#     low = x & 0x0F
#     high = (x >> 4) & 0x0F
#     unpacked = torch.stack([low, high], dim=-1).view(*x.shape[:-1], -1)  # 在最后添加新维度
    
#     # 将解包后的数据展平以便处理
#     original_shape = unpacked.shape
#     unpacked_flat = unpacked.view(-1)
    
#     # 提取符号位、指数码、尾数位
#     s = (unpacked_flat >> 3) & 0x01
#     e_code = (unpacked_flat >> 1) & 0x03
#     m = unpacked_flat & 0x01
    
#     # 转换为浮点数进行计算
#     s = s.float()
#     e_code = e_code.int()
#     m = m.float()
    
#     # 计算实际数值
#     val_subnormal = m * 0.25  # 0.5（尾数） * 2^-1（指数）
#     exponent = (e_code - 1).float()
#     val_normal = (1.0 + m * 0.5) * torch.pow(2.0, exponent)
    
#     # 根据指数码选择数值
#     e_code_zero = (e_code == 0)
#     val = torch.where(e_code_zero, val_subnormal, val_normal)
    
#     # 应用符号位
#     val = val * torch.where(s > 0.5, -1.0, 1.0)
    
#     # 转换为float8_e4m3fn并恢复形状
#     val_fp8 = val.to(torch.float8_e4m3fn)
#     val_fp8 = val_fp8.view(original_shape)
    
#     return val_fp8


# # 示例输入：两个uint8数值，0b00010001（17）和0b00110011（51）
# x = torch.tensor([17, 51], dtype=torch.uint8)
# result = unpack_uint8_to_fp4(x)
# print(result)  # 应输出四个FP8数值：[0.25, 0.25, 1.5, 1.5]

# import torch

# sorted_candidates = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# def quantize_to_fp4_e2m1(x_scaled):
#     # 计算绝对值
#     abs_x = torch.abs(x_scaled)
#     # 找到插入位置
#     indices = torch.bucketize(abs_x, sorted_candidates)
#     # 确保索引不越界
#     indices = torch.clamp(indices, 0, len(sorted_candidates) - 1)
#     # 左鄰居索引
#     left_idx = torch.clamp(indices - 1, 0)
#     # 获取左右候选值
#     left_val = sorted_candidates[left_idx]
#     right_val = sorted_candidates[indices]
#     # 计算差异
#     left_diff = torch.abs(abs_x - left_val)
#     right_diff = torch.abs(abs_x - right_val)
#     # 选择更接近的候选值
#     mask = left_diff < right_diff
#     selected_val = torch.where(mask, left_val, right_val)
#     # 处理超出最大值的情况
#     max_val = sorted_candidates[-1]
#     selected_val = torch.where(abs_x > max_val, max_val, selected_val)
#     # 应用符号
#     quantized = selected_val * torch.sign(x_scaled)
#     return quantized

# input_scale = torch.tensor(0.5, dtype=torch.float32)
# x = torch.randn(2, 2, dtype=torch.float16)

# # 将输入转换为 float32 并应用缩放
# x_float32 = x.float()
# x_scaled = x_float32 / input_scale
# print(x)
# # 量化到 FP4
# x_quantized = quantize_to_fp4_e2m1(x_scaled)
# print(x_quantized)
# # 反量化并转换为 float16
# x_dequantized = x_quantized * input_scale
# # x_dequantized_fp16 = x_dequantized.to(torch.float16)
# print(x_dequantized.dtype)
# from safetensors import safe_open
# import torch

# def load_specified_linear_weights():
#     ckpt_path = 'model-00001-of-000163.safetensors'  # noqa
#     layer_id = 0
#     # prefix = f'model.layers.{layer_id}.self_attn.q_proj.'
#     prefix = f'model.layers.{layer_id}.mlp.down_proj.'
#     keys = ['weight', 'weight_scale_inv']
#     tensors = {}
#     with safe_open(ckpt_path, framework='pt', device='cuda') as f:
#         for key in keys:
#             tensors[key] = f.get_tensor(prefix + key)

#     return tensors['weight'], tensors['weight_scale_inv']

# import torch

# def dequantize_weights_torch(quantized_weights: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
#     # 确认尺寸匹配
#     H, W = quantized_weights.shape
#     assert H % 128 == 0 and W % 128 == 0, "权重矩阵尺寸必须是 128 的倍数"
#     assert scale_matrix.shape == (H // 128, W // 128), "scale 矩阵尺寸不匹配"
#     scale_expanded = torch.kron(
#         scale_matrix,
#         torch.ones((128, 128), dtype=scale_matrix.dtype, device=scale_matrix.device)
#     )

#     dequantized = quantized_weights.float() * scale_expanded.float()
#     return dequantized.half()

# weight, weight_scale = load_specified_linear_weights()
# print(weight, weight_scale)
# dequant_weight = dequantize_weights_torch(weight, weight_scale)
# print(dequant_weight, dequant_weight.shape)