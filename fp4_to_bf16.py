import torch
import torch.nn as nn
from safetensors import safe_open
import tensorrt_llm
import unittest
from parameterized import parameterized

torch.manual_seed(0)

def load_specified_linear_weights():
    ckpt_path = 'model-00001-of-000163.safetensors'  # noqa
    layer_id = 0
    prefix = f'model.layers.{layer_id}.mlp.down_proj.'
    keys = ['weight', 'weight_scale_inv']
    tensors = {}
    with safe_open(ckpt_path, framework='pt', device='cuda:0') as f:
        for key in keys:
            tensors[key] = f.get_tensor(prefix + key)

    return tensors['weight'], tensors['weight_scale_inv']

def dequantize_weights_torch(quantized_weights: torch.Tensor, scale_matrix: torch.Tensor) -> torch.Tensor:
    H, W = quantized_weights.shape
    assert H % 128 == 0 and W % 128 == 0
    assert scale_matrix.shape == (H // 128, W // 128)
    scale_expanded = torch.kron(
        scale_matrix,
        torch.ones((128, 128), dtype=scale_matrix.dtype, device=scale_matrix.device)
    )

    dequantized = quantized_weights.float() * scale_expanded.float()
    return dequantized.half()

# weight, weight_scale = load_specified_linear_weights()
# weight_ref = dequantize_weights_torch(weight, weight_scale)


# # group_size_awq = 64
# batch_size = 16384

# # qweight_ref, qzeros_ref, scales_ref = load_specified_linear_weights('/root/turbomind/model-00001-of-00074.safetensors')
# # 18432
# in_features = weight_ref.shape[1]
# # 7168
# out_features = weight_ref.shape[0]

# x = torch.randn((batch_size, in_features),
#                 device=weight_ref.device,
#                 dtype=torch.float16) * 0.1

# # weight_awq = dequantize(qweight_ref, qzeros_ref, scales_ref, group_size_awq)
# print(f'-- dequantization: weight_ref.shape={weight_ref.shape}, weight_ref: \n{weight_ref}')
# linear_ref = nn.Linear(in_features, out_features, bias=False, device='cuda:0')
# with torch.no_grad():
#     linear_ref.weight = nn.Parameter(weight_ref)
#     res_ref = linear_ref(x)
#     print(linear_ref.weight.shape)
#     print(f'nn.linear.ref_res: {res_ref}, {res_ref.shape}')


def load_weight(path):
    file_path = path
    layer_id = 0
    prefix = f'model.layers.{layer_id}.mlp.down_proj.'
    keys = ['input_scale', 
            'weight', 
            'weight_scale', 
            'weight_scale_2']
    tensors = {}

    with safe_open(file_path, framework="pt", device="cuda:0") as f:
        for key in keys:
            tensor = f.get_tensor(prefix + key)
            tensors[key] = tensor
            # print(f"Tensor '{key}': shape={tensor.shape}, dtype={tensor.dtype}")
    return tensors

def unpack_uint8_to_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8, "Input tensor must be of type torch.uint8"
    
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).view(*x.shape[:-1], -1)
    
    original_shape = unpacked.shape
    unpacked_flat = unpacked.view(-1)
    
    s = (unpacked_flat >> 3) & 0x01
    e_code = (unpacked_flat >> 1) & 0x03
    m = unpacked_flat & 0x01
    
    s = s.float()
    e_code = e_code.int()
    m = m.float()
    
    val_subnormal = m * 0.5  # 0.5（尾数） * 2^-1（指数）
    exponent = (e_code - 1).float()
    val_normal = (1.0 + m * 0.5) * torch.pow(2.0, exponent)
    
    e_code_zero = (e_code == 0)
    val = torch.where(e_code_zero, val_subnormal, val_normal)
    
    val = val * torch.where(s > 0.5, -1.0, 1.0)
    
    val_fp8 = val.to(torch.float8_e4m3fn)
    val_fp8 = val_fp8.view(original_shape)
    
    return val_fp8


def dequantize_w(w, w_block_scale, w_global_scale, group_size):

    w = unpack_uint8_to_fp4(w)
    w = w.float()
    # print(w)
    w_block_scale = w_block_scale.float()

    for i in range(w_block_scale.shape[-1]):
        start = i * group_size
        end = start + group_size
        w[:, start:end] = (w[:, start:end] * w_block_scale[:, i:i + 1])
    
    w = w * w_global_scale

    return w.half()

def dequantize_x(x, x_block_scale, x_global_scale, group_size):
    # x = unpack_uint8_to_fp4(x)
    # x = x.float()
    x_block_scale = x_block_scale.float()

    for i in range(x_block_scale.shape[-1]):
        start = i * group_size
        end = start + group_size
        x[:, start:end] = (x[:, start:end] * x_block_scale[:, i:i + 1])
    
    x = x * x_global_scale

    return x


sorted_candidates = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32).to("cuda:0")

def quantize_to_fp4_e2m1(x : torch.Tensor, x_global_scale, group_size):
    org_shape = x.shape
    # [n_group, group_size]
    x = x.reshape(-1, group_size)
    # [n_group, 1]
    max_val = x.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)

    x_block_scale = max_val / 6.0
    x_fp4 = x / x_block_scale
    # 计算绝对值
    abs_x = torch.abs(x_fp4)
    # 找到插入位置
    indices = torch.bucketize(abs_x, sorted_candidates)
    # 确保索引不越界
    indices = torch.clamp(indices, 0, len(sorted_candidates) - 1)
    # 左鄰居索引
    left_idx = torch.clamp(indices - 1, 0)
    # 获取左右候选值
    left_val = sorted_candidates[left_idx]
    right_val = sorted_candidates[indices]
    # 计算差异
    left_diff = torch.abs(abs_x - left_val)
    right_diff = torch.abs(abs_x - right_val)
    # 选择更接近的候选值
    mask = left_diff < right_diff
    selected_val = torch.where(mask, left_val, right_val)
    # 处理超出最大值的情况
    max_val = sorted_candidates[-1]
    selected_val = torch.where(abs_x > max_val, max_val, selected_val)
    # 应用符号
    x_fp4 = selected_val * torch.sign(x_fp4)
    # use global_scale to quantize block_scale
    x_block_scale = torch.clamp((x_block_scale / x_global_scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    # [ci, co/group_size]
    x_block_scale = x_block_scale.view(org_shape[0], -1)

    x_fp4 = x_fp4.reshape(org_shape)
    return x_fp4, x_block_scale
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

# tensors = load_weight('/root/turbomind/model-00001-of-00080.safetensors')
# input_global_scale = tensors['input_scale']
# weight_fp4 = tensors['weight']
# weight_block_scale = tensors['weight_scale']
# weight_global_scale = tensors['weight_scale_2']

# group_size = 16
# batch_size = 16384
# # 7186
# in_features = tensors['weight'].shape[0]
# # 18432
# out_features = tensors['weight'].shape[1]*2

# input = x
# # print(input)
# input_fp4, input_block_scale = quantize_to_fp4_e2m1(input, input_global_scale, group_size)
# # print(input_fp4, input_block_scale)
# input = dequantize_x(input_fp4, input_block_scale, input_global_scale, group_size)
# # print(input)

# weight = dequantize_w(weight_fp4, weight_block_scale, weight_global_scale, group_size)
# print(f'-- dequantization: weight.shape={weight.shape}, weight: \n{weight}')

# fp4_linear = nn.Linear(out_features, in_features, bias=False, device='cuda:0')
# with torch.no_grad():
#     # print(fp4_linear.weight.shape)
#     fp4_linear.weight = nn.Parameter(weight)
#     # print(fp4_linear.weight.shape)
#     fp4_res = fp4_linear(input)
#     print(f'nn.linear.res: {fp4_res}, {fp4_res.shape}')

# Used by the (fp16 -> int4) quant layer + int4 gemm network.

def e2m1_and_ufp8_scale_to_float_tensor_v2(
    e2m1_tensor: torch.Tensor,
    ufp8_scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size,
    ufp8_type: int = 1,
):
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size,
        ufp8_type)
    return float_tensor


class TestFunctional(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    @parameterized.expand(list([[1024, 1024, torch.half, False],
                            [2, 512, torch.bfloat16, False],
                            [13, 16, torch.half, True]]))
    def test_fp4_quantize_torch(self, m, k, dtype, use_ue8m0):
        a = torch.randn([m, k], dtype=torch.float32).to(dtype).float()
        a_global_sf = (448 * 6) / a.abs().max().float()
        sf_vec_size = 16

        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(
            a.to(dtype).cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0)

        print(unpack_uint8_to_fp4(a_fp4))
        print(a_sf)
        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                        1 / a_global_sf,
                                                        sf_vec_size)
        
        input_fp4, input_block_scale = quantize_to_fp4_e2m1(a.to(dtype).cuda(), (1/a_global_sf).cuda(), sf_vec_size)
        print(input_fp4, input_block_scale)
        input = dequantize_x(input_fp4, input_block_scale, (1/a_global_sf).cuda(), sf_vec_size)
        # my_a_pt = dequantize_w(a_fp4, a_sf, (1 / a_global_sf).cuda(), sf_vec_size)

        torch.cuda.synchronize()
        if not use_ue8m0:
            # The gap is too large for ue8m0, so we just make sure that it runs
            self.assertTrue(torch.allclose(a, input.float().cpu(), atol=1, rtol=1))

    @parameterized.expand(
        list([
            [1024, 1024, 1024],
            [7, 32, 32],
        ])
    )
    # @skip_pre_blackwell_unittest
    def test_fp4_quantize_gemm_torch(self, m, n, k):
        # pytest.skip("https://nvbugs/5100633")
        a = torch.randn([m, k], dtype=torch.float32)
        b = torch.randn([n, k], dtype=torch.float32)
        a_global_sf = (448 * 6) / a.abs().max().float()
        b_global_sf = (448 * 6) / b.abs().max().float()
        ab_global_sf = 1 / (a_global_sf * b_global_sf)
        ab_global_sf = ab_global_sf.cuda()
        sf_vec_size = 16
        a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.half().cuda(),
                                                    a_global_sf.cuda(),
                                                    sf_vec_size, False)
        print(unpack_uint8_to_fp4(a_fp4).shape)
        # torch.Size([1024, 1024])
        print(a_sf.shape)
        # torch.Size([65536])
        print(a_sf)
        # tensor([116, 114, 116,  ..., 115, 111, 115], device='cuda:0',
        # dtype=torch.uint8)
        b_fp4, b_sf = torch.ops.trtllm.fp4_quantize(b.half().cuda(),
                                                    b_global_sf.cuda(),
                                                    sf_vec_size, False)

        a_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(a_fp4.cpu(), a_sf.cpu(),
                                                      1 / a_global_sf,
                                                      sf_vec_size)
        b_pt = e2m1_and_ufp8_scale_to_float_tensor_v2(b_fp4.cpu(), b_sf.cpu(),
                                                      1 / b_global_sf,
                                                      sf_vec_size)
        # print(a_pt)
        # c = (torch.ops.trtllm.fp4_gemm(a_fp4, b_fp4, a_sf, b_sf, ab_global_sf,
        #                                False).float().cpu())
        my_a_fp4, my_a_sf = quantize_to_fp4_e2m1(a.half().cuda(), (1/a_global_sf).cuda(), sf_vec_size)
        print(my_a_fp4.shape)
        # torch.Size([1024, 1024])
        print(my_a_sf.shape)
        # torch.Size([1024, 64])
        print(my_a_sf)
        # tensor([[192., 160., 192.,  ..., 256., 192., 144.],
        # [160., 192., 240.,  ..., 120., 192., 176.],
        # [144., 192., 176.,  ..., 192., 208., 224.],
        # ...,
        # [192., 240., 176.,  ..., 176., 144., 256.],
        # [288., 176., 192.,  ..., 192., 112., 120.],
        # [192., 208., 240.,  ..., 176., 120., 176.]], device='cuda:0',
        # dtype=torch.float8_e4m3fn)
        my_b_fp4, my_b_sf = quantize_to_fp4_e2m1(b.half().cuda(), (1/b_global_sf).cuda(), sf_vec_size)        
        # print(my_a_fp4)
        # print(unpack_uint8_to_fp4(a_fp4).float())
        # abs_diff = torch.abs(unpack_uint8_to_fp4(a_fp4).float() - my_a_fp4).float()
        # print(abs_diff)
        # rel_diff = abs_diff / (torch.max(torch.abs(unpack_uint8_to_fp4(a_fp4).float()), torch.abs(my_a_fp4)) + 1e-08)
        # abs_diff = torch.sum(abs_diff) / abs_diff.numel()
        # rel_diff = torch.sum(rel_diff) / rel_diff.numel()
        my_a = dequantize_x(my_a_fp4, my_a_sf, (1/a_global_sf).cuda(), sf_vec_size)
        my_b = dequantize_x(my_b_fp4, my_b_sf, (1/b_global_sf).cuda(), sf_vec_size)
        # print(my_a.float())
        torch.cuda.synchronize()
        c = torch.nn.functional.linear(a, b)
        c_pt = torch.nn.functional.linear(a_pt, b_pt)
        my_c = torch.nn.functional.linear(my_a.float(), my_b.float())
        my_c = my_c.cpu()
        my_a = my_a.cpu()
        my_b = my_b.cpu()
        abs_diff = torch.abs(c_pt - my_c).float()
        rel_diff = abs_diff / (torch.max(torch.abs(c_pt), torch.abs(my_c)) + 1e-08)
        abs_diff = torch.sum(abs_diff) / abs_diff.numel()
        rel_diff = torch.sum(rel_diff) / rel_diff.numel()
        # print(c)
        # print(c_pt)
        # print(my_c)
        # print(abs_diff)
        # print(rel_diff)
        # print(calc_diff(my_c, c))

        # diff of a_fp4 and my_a_fp4: tensor(0.0449) tensor(0.0163)

        # diff of a, a_pt, my_a:
        # diff of a and my_a: tensor(0.0726) tensor(0.1664)
        # diff of a and a_pt: tensor(0.0714) tensor(0.1651)
        # diff of a_pt and my_a: tensor(0.0150) tensor(0.0164)

        # diff of b_pt and my_b: tensor(0.0149) tensor(0.0164)

        # diff of c_pt and my_c: tensor(2.7666) tensor(0.2067) sim_diff:0.0059
        # diff of c and c_pt: tensor(3.4306) tensor(0.2399) sim_diff:0.0090
        # diff of c and my_c: tensor(3.5227) tensor(0.2439) sim_diff:0.0095
        self.assertTrue(torch.allclose(c_pt, my_c.float().cpu(), atol=1e-2, rtol=1e-2))


# abs_diff = torch.abs(fp4_res - res_ref).float()
# rel_diff = abs_diff / torch.max(torch.abs(res_ref), torch.abs(fp4_res))
# rtol = 0.01
# atol = 0.0001
# outliers = abs_diff > atol + rtol * torch.abs(res_ref)
# abs_diff = torch.sum(abs_diff) / abs_diff.numel()
# rel_diff = torch.sum(rel_diff) / rel_diff.numel()
# outliers = torch.sum(outliers) / outliers.shape[0]
# print(f'abs_diff {abs_diff:4f}, '
#       f'rel_diff {rel_diff:4f}, '
#       f'outliers {outliers:4f}')

# abs_diff 0.004328, rel_diff 0.239396, outliers 6722.556641