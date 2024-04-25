
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
#     size_hints=[16384, 4096],
#     reduction_hint=ReductionHint.OUTER,
#     filename=__file__,
#     meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
# )
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14336
    rnumel = 2634
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (2634*x1)
        tmp1 = tl.full([1, 1], 1179648, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (2634*x1)) % 1179648))), rmask & tmp2, other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.where(tmp2, tmp4, 0)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp6, 0)
        tmp8 = 1.0
        tmp9 = tl.where(tmp2, tmp8, 0)
        tmp10 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp11 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(rmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
    tl.store(out_ptr2 + (x3), tmp15, None)


def get_args():
    arg_0 = rand_strided((128, 32, 96, 96), (294912, 1, 3072, 32), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 32, 1, 1, 448), (14336, 1, 14336, 14336, 32), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 32, 1, 1, 448), (14336, 1, 14336, 14336, 32), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 32, 1, 1, 448), (14336, 1, 14336, 14336, 32), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    # with torch.xpu._DeviceGuard(0):
    #     torch.xpu.set_device(0)
    #     stream0 = get_xpu_stream(0)
    grid=lambda meta: (14336, )
    triton_red_fused__native_batch_norm_legit_functional_2[grid](*args, 14336, 2634, 1, 1024)


# def benchmark_all_configs(args):
#     with torch.xpu._DeviceGuard(0):
#         torch.xpu.set_device(0)
#         return triton_red_fused__native_batch_norm_legit_functional_2.benchmark_all_configs(*args, 14336, 2634, grid=grid(14336))


if __name__ == '__main__':
    # from torch._inductor.utils import get_num_bytes
    # from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    call(args)
    # ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    # num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    # gb_per_s = num_gb / (ms / 1e3)
    # print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")