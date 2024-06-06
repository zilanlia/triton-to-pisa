# kernel path: /data/liyang/pytorch/inductor_log/timm_models/beit_base_patch16_224/amp_bf16/xc/cxcjcdhivwbp24x7aveybyuqttyaxf2s4cnojoueqps57ogodqud.py
# Source Nodes: [scaled_dot_product_attention], Original ATen: [aten._softmax, aten.add]
# scaled_dot_product_attention => add_2, amax, convert_element_type_8, convert_element_type_9, div, exp, sub_1, sum_1

import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@persistent_reduction(
    size_hints=[262144, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_add_6(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2364
    tmp0 = tl.load(in_ptr0 + (r2 + (197*x3)), rmask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (197*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, float("-inf"))
    tmp7 = triton_helpers.max2(tmp6, 1)[:, None]
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = tmp9 / tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (197*x3)), tmp15, rmask & xmask)


def get_args():
    arg_0 = rand_strided((768, 197, 197), (38809, 197, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 12, 197, 197), (465708, 38809, 197, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((64, 12, 197, 197), (465708, 38809, 197, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax_add_6.run(*args, 151296, 197, grid=grid(151296), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax_add_6.benchmark_all_configs(*args, 151296, 197, grid=grid(151296))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
