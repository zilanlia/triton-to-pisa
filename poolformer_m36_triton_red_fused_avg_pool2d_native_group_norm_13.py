# kernel path: /data/liyang/pytorch/inductor_log/timm_models/poolformer_m36/amp_bf16/uq/cuqmzonifwqufpasl54wrkxkkq2pvbydznzy6pac3xano74vez4l.py
# Source Nodes: [getattr_getattr_l__self___stages___0___blocks___1___token_mixer_pool, group_norm_3], Original ATen: [aten.avg_pool2d, aten.native_group_norm]
# getattr_getattr_l__self___stages___0___blocks___1___token_mixer_pool => avg_pool2d_1
# group_norm_3 => var_mean_3

import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@reduction(
    size_hints=[256, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_avg_pool2d_native_group_norm_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_avg_pool2d_native_group_norm_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    x6 = xindex % 4
    tmp87_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp87_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp87_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 56) % 56
        r1 = rindex % 56
        r5 = rindex
        tmp80 = tl.load(in_ptr1 + (r5 + (75264*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp81 = tl.load(in_ptr0 + (r5 + (75264*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.load(in_ptr2 + ((24*x6) + (r5 // 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = (-1) + r2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 56, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tmp2 & tmp4
        tmp6 = (-1) + r1
        tmp7 = tmp6 >= tmp1
        tmp8 = tmp6 < tmp3
        tmp9 = tmp7 & tmp8
        tmp10 = tmp5 & tmp9
        tmp11 = tl.load(in_ptr0 + ((-57) + r5 + (75264*x0)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.where(tmp10, tmp11, 0.0)
        tmp13 = r1
        tmp14 = tmp13 >= tmp1
        tmp15 = tmp13 < tmp3
        tmp16 = tmp14 & tmp15
        tmp17 = tmp5 & tmp16
        tmp18 = tl.load(in_ptr0 + ((-56) + r5 + (75264*x0)), rmask & tmp17 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.where(tmp17, tmp18, 0.0)
        tmp20 = tmp19 + tmp12
        tmp21 = 1 + r1
        tmp22 = tmp21 >= tmp1
        tmp23 = tmp21 < tmp3
        tmp24 = tmp22 & tmp23
        tmp25 = tmp5 & tmp24
        tmp26 = tl.load(in_ptr0 + ((-55) + r5 + (75264*x0)), rmask & tmp25 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.where(tmp25, tmp26, 0.0)
        tmp28 = tmp27 + tmp20
        tmp29 = r2
        tmp30 = tmp29 >= tmp1
        tmp31 = tmp29 < tmp3
        tmp32 = tmp30 & tmp31
        tmp33 = tmp32 & tmp9
        tmp34 = tl.load(in_ptr0 + ((-1) + r5 + (75264*x0)), rmask & tmp33 & xmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.where(tmp33, tmp34, 0.0)
        tmp36 = tmp35 + tmp28
        tmp37 = tmp32 & tmp16
        tmp38 = tl.load(in_ptr0 + (r5 + (75264*x0)), rmask & tmp37 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.where(tmp37, tmp38, 0.0)
        tmp40 = tmp39 + tmp36
        tmp41 = tmp32 & tmp24
        tmp42 = tl.load(in_ptr0 + (1 + r5 + (75264*x0)), rmask & tmp41 & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tl.where(tmp41, tmp42, 0.0)
        tmp44 = tmp43 + tmp40
        tmp45 = 1 + r2
        tmp46 = tmp45 >= tmp1
        tmp47 = tmp45 < tmp3
        tmp48 = tmp46 & tmp47
        tmp49 = tmp48 & tmp9
        tmp50 = tl.load(in_ptr0 + (55 + r5 + (75264*x0)), rmask & tmp49 & xmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.where(tmp49, tmp50, 0.0)
        tmp52 = tmp51 + tmp44
        tmp53 = tmp48 & tmp16
        tmp54 = tl.load(in_ptr0 + (56 + r5 + (75264*x0)), rmask & tmp53 & xmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.where(tmp53, tmp54, 0.0)
        tmp56 = tmp55 + tmp52
        tmp57 = tmp48 & tmp24
        tmp58 = tl.load(in_ptr0 + (57 + r5 + (75264*x0)), rmask & tmp57 & xmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tl.where(tmp57, tmp58, 0.0)
        tmp60 = tmp59 + tmp56
        tmp61 = 1.0
        tmp62 = tl.where(tmp10, tmp61, 0.0)
        tmp63 = tl.where(tmp17, tmp61, 0.0)
        tmp64 = tmp63 + tmp62
        tmp65 = tl.where(tmp25, tmp61, 0.0)
        tmp66 = tmp65 + tmp64
        tmp67 = tl.where(tmp33, tmp61, 0.0)
        tmp68 = tmp67 + tmp66
        tmp69 = tl.where(tmp37, tmp61, 0.0)
        tmp70 = tmp69 + tmp68
        tmp71 = tl.where(tmp41, tmp61, 0.0)
        tmp72 = tmp71 + tmp70
        tmp73 = tl.where(tmp49, tmp61, 0.0)
        tmp74 = tmp73 + tmp72
        tmp75 = tl.where(tmp53, tmp61, 0.0)
        tmp76 = tmp75 + tmp74
        tmp77 = tl.where(tmp57, tmp61, 0.0)
        tmp78 = tmp77 + tmp76
        tmp79 = tmp60 / tmp78
        tmp82 = tmp79 - tmp81
        tmp84 = tmp82 * tmp83
        tmp85 = tmp80 + tmp84
        tmp86 = tl.broadcast_to(tmp85, [XBLOCK, RBLOCK])
        tmp87_mean_next, tmp87_m2_next, tmp87_weight_next = triton_helpers.welford_reduce(
            tmp86, tmp87_mean, tmp87_m2, tmp87_weight,
        )
        tmp87_mean = tl.where(rmask & xmask, tmp87_mean_next, tmp87_mean)
        tmp87_m2 = tl.where(rmask & xmask, tmp87_m2_next, tmp87_m2)
        tmp87_weight = tl.where(rmask & xmask, tmp87_weight_next, tmp87_weight)
        tl.store(out_ptr0 + (r5 + (75264*x0)), tmp79, rmask & xmask)
    tmp87_tmp, tmp88_tmp, tmp89_tmp = triton_helpers.welford(
        tmp87_mean, tmp87_m2, tmp87_weight, 1
    )
    tmp87 = tmp87_tmp[:, None]
    tmp88 = tmp88_tmp[:, None]
    tmp89 = tmp89_tmp[:, None]
    tl.store(out_ptr1 + (x0), tmp87, xmask)
    tmp98_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp98_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp98_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        tmp90 = tl.load(in_ptr1 + (r5 + (75264*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp91 = tl.load(out_ptr0 + (r5 + (75264*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp92 = tl.load(in_ptr0 + (r5 + (75264*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp94 = tl.load(in_ptr2 + ((24*x6) + (r5 // 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = tmp91 - tmp92
        tmp95 = tmp93 * tmp94
        tmp96 = tmp90 + tmp95
        tmp97 = tl.broadcast_to(tmp96, [XBLOCK, RBLOCK])
        tmp98_mean_next, tmp98_m2_next, tmp98_weight_next = triton_helpers.welford_reduce(
            tmp97, tmp98_mean, tmp98_m2, tmp98_weight,
        )
        tmp98_mean = tl.where(rmask & xmask, tmp98_mean_next, tmp98_mean)
        tmp98_m2 = tl.where(rmask & xmask, tmp98_m2_next, tmp98_m2)
        tmp98_weight = tl.where(rmask & xmask, tmp98_weight_next, tmp98_weight)
    tmp98_tmp, tmp99_tmp, tmp100_tmp = triton_helpers.welford(
        tmp98_mean, tmp98_m2, tmp98_weight, 1
    )
    tmp98 = tmp98_tmp[:, None]
    tmp99 = tmp99_tmp[:, None]
    tmp100 = tmp100_tmp[:, None]
    tl.store(out_ptr2 + (x0), tmp99, xmask)
    tl.store(out_ptr3 + (x0), tmp100, xmask)


def get_args():
    arg_0 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((96, 1, 1), (1, 1, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 96, 56, 56), (301056, 3136, 56, 1), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((64, 1, 1, 1, 4), (4, 256, 256, 256, 1), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((64, 1, 1, 1, 4), (4, 256, 256, 256, 1), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((64, 1, 1, 1, 4), (4, 256, 256, 256, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_avg_pool2d_native_group_norm_13.run(*args, 256, 75264, grid=grid(256), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_avg_pool2d_native_group_norm_13.benchmark_all_configs(*args, 256, 75264, grid=grid(256))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
