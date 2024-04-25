
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
#     size_hints=[4096, 4096],
#     reduction_hint=ReductionHint.OUTER,
#     filename=__file__,
#     meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_hardswish_backward_native_batch_norm_backward_67', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
# )
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_hardswish_backward_native_batch_norm_backward_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3584
    rnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 8
    x1 = (xindex // 8)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (8*r2) + (28672*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (x0 + (8*r2) + (28672*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (x0 + (8*r2) + (28672*x1)), rmask & xmask, other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = -3.0
        tmp3 = tmp1 < tmp2
        tmp4 = 3.0
        tmp5 = tmp1 <= tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp1 / tmp4
        tmp9 = 0.5
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp5, tmp11, tmp7)
        tmp13 = 0.0
        tmp14 = tl.where(tmp3, tmp13, tmp12)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp14 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)


def get_args():
    arg_0 = rand_strided((128, 8, 112, 112), (100352, 1, 896, 8), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 8, 112, 112), (100352, 1, 896, 8), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((128, 8, 112, 112), (100352, 1, 896, 8), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((1, 8, 1, 1), (8, 1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((8, 448), (1, 8), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((8, 448), (1, 8), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    # with torch.xpu._DeviceGuard(0):
    #     torch.xpu.set_device(0)
    #     stream0 = get_xpu_stream(0)
    grid=lambda meta: (3584, )
    triton_red_fused__native_batch_norm_legit_functional_hardswish_backward_native_batch_norm_backward_67[grid](*args, 3584, 3584, 1, 2048)


# def benchmark_all_configs(args):
#     with torch.xpu._DeviceGuard(0):
#         torch.xpu.set_device(0)
#         return triton_red_fused__native_batch_norm_legit_functional_hardswish_backward_native_batch_norm_backward_67.benchmark_all_configs(*args, 3584, 3584, grid=grid(3584))


if __name__ == '__main__':
    # from torch._inductor.utils import get_num_bytes
    # from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    call(args)
    # ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    # num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    # gb_per_s = num_gb / (ms / 1e3)
    # print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")