# turbomind

功能实现在turbomind/fp4_to_bf16.py

处理三种不同的权重类型：
* 原始权重(fp16类型，包含一个weight Tensor)
* 量化权重(fp4类型，包含weight(fp4)和weight_scale(fp8)两个Tensor，以及一个fp32类型的per-tensor global_scale)
* pack量化权重(fp4类型，但以uint8格式存储，包含weight(uint8)和weight_scale(fp8)两个Tensor，以及一个fp32类型的per-tensor global_scale)

deepseek-fp4使用的量化数据格式为在pack量化权重基础上引入动态双重量化，额外将global_scale拆分为input_scale和weight_scale_2两个fp32类型的per-tensor scale，反量化时将这两个scale相乘作为global_scale即可

## 函数功能：

### `unpack_uint8_to_fp4(x: torch.Tensor)`

接受一个pack量化权重的uint8类型weight，输出相应的量化权重的fp4类型weight

### `dequantize_unpack(w, w_block_scale, w_global_scale, group_size)`

接受一组pack量化权重及相应的group size(一般取16)，输出fp32类型的原始权重

### `dequantize(x, x_block_scale, x_global_scale, group_size)`

接受一组量化权重及相应的group size(一般取16)，输出fp32类型的原始权重

### `quantize_to_fp4_e2m1(x : torch.Tensor, x_global_scale, group_size)`

接受一个原始权重Tensor及预设的global_scale和相应的group_size(一般取16)，输出相应的量化权重的fp4类型weight和fp8类型weight_scale

### `e2m1_and_ufp8_scale_to_float_tensor_v2(e2m1_tensor, ufp8_scale_tensor, global_scale_tensor, sf_vec_size, ufp8_type = 1)`

对tensorrt_llm实现的包装，用作reference
接受一组pack量化权重及相应的group size(一般取16)，输出fp32类型的原始权重

### `torch.ops.trtllm.fp4_quantize`

tensorrt_llm实现，用作reference
接受一个原始权重Tensor及预设的global_scale和相应的group_size(一般取16)，输出相应的量化权重的fp4类型weight和fp8类型weight_scale

### `test_fp4_quantize_torch`

* 生成随机权重`a`及量化用的global_scale：`a_global_sf`
* 将`a`量化为`a_fp4`及`a_sf`
* 将`a_fp4`及`a_sf`反量化为`a_result`
* 对比`a`与`a_result`

### `test_fp4_quantize_gemm_torch`

* 生成随机权重`a`、`b`及量化用的global_scale：`a_global_sf`、`b_global_sf`
* 使用tensorrt_llm实现将`a`、`b`量化为`a_fp4`及`a_sf`、`b_fp4`及`b_sf`
* 将`a_fp4`及`a_sf`、`b_fp4`及`b_sf`反量化为`a_pt`、`b_pt`
* 使用自己的实现将`a`、`b`量化为`my_a_fp4`及`my_a_sf`、`my_b_fp4`及`my_b_sf`
* 将`my_a_fp4`及`my_a_sf`、`my_b_fp4`及`my_b_sf`反量化为`my_a`、`my_b`
* 计算`c_pt = a_pt * b_pt`、`my_c = my_a * my_b`
* 对比`c_pt`与`my_c`
