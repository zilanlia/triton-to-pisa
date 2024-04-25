
import triton
import triton.language as tl
# from torch._inductor.ir import ReductionHint
# from torch._inductor.ir import TileHint
# from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
# from torch._inductor.utils import instance_descriptor
import triton_helpers
import intel_extension_for_pytorch 
from helper import rand_strided
import torch
# from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
# from torch._inductor.triton_heuristics import grid

# @reduction(
#     size_hints=[256, 524288],
#     reduction_hint=ReductionHint.INNER,
#     filename=__file__,
#     meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_113', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]}
# )
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_113(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr5 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp7 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp7 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tmp27 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tmp17 * tmp27
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr3 + (x0), tmp28, xmask)
    tl.store(out_ptr4 + (x0), tmp30, xmask)


def get_args():
    arg_0 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 256, 56, 56), (802816, 3136, 56, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_6 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_7 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_8 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_9 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_10 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_11 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_12 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    arg_13 = rand_strided((256,), (1,), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13,


def call(args):
    # with torch.xpu._DeviceGuard(0):
    #     torch.xpu.set_device(0)
    #     stream0 = get_xpu_stream(0)
    grid=lambda meta: (256, )
    triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_113[grid](*args, 256, 401408, 1, 256)


# def benchmark_all_configs(args):
#     with torch.xpu._DeviceGuard(0):
#         torch.xpu.set_device(0)
#         return triton_red_fused__native_batch_norm_legit_functional_add_native_batch_norm_backward_threshold_backward_113.benchmark_all_configs(*args, 256, 401408, grid=grid(256))


if __name__ == '__main__':
    # from torch._inductor.utils import get_num_bytes
    # from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    call(args)
    # ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    # num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    # gb_per_s = num_gb / (ms / 1e3)
    # print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")