# kernel path: /data/liyang/pytorch/inductor_log/timm_models/gmixer_24_224/amp_bf16/hy/chy7k6h7cfz4r7evjgyw2q6ewgjyyekr52ccs4afb2ptk3crhcl3.py
# Source Nodes: [add_4, add_5, add_6, getattr_l__self___blocks___2___mlp_channels_drop2, getattr_l__self___blocks___3___norm2], Original ATen: [aten.add, aten.clone, aten.native_layer_norm]
# add_4 => add_17
# add_5 => add_20
# add_6 => add_24
# getattr_l__self___blocks___2___mlp_channels_drop2 => clone_17
# getattr_l__self___blocks___3___norm2 => add_25, add_26, clone_21, convert_element_type_59, convert_element_type_60, mul_28, mul_29, rsqrt_7, sub_7, var_mean_7

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
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_clone_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask & xmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp12 = tl.load(in_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp9
        tmp21 = 384.0
        tmp22 = tmp10 / tmp21
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp26 * tmp27
        tmp30 = tmp28 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp31, rmask & xmask)


def get_args():
    arg_0 = rand_strided((128, 196, 384), (75264, 384, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((49152, 196), (196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((25088, 384), (384, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((49152, 196), (196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((384,), (1,), device='xpu:0', dtype=torch.float32)
    arg_5 = rand_strided((384,), (1,), device='xpu:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 196, 384), (75264, 384, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused_add_clone_native_layer_norm_10.run(*args, 25088, 384, grid=grid(25088), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused_add_clone_native_layer_norm_10.benchmark_all_configs(*args, 25088, 384, grid=grid(25088))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
