# kernel path: /data/liyang/pytorch/inductor_log/timm_models/cait_m36_384/amp_bf16/lm/clmfmbbu7pyjknoweqzm3glnbe4qzqwoltrvnkeaavapqa2wghpo.py
# Source Nodes: [softmax], Original ATen: [aten._softmax]
# softmax => amax, clone_2, convert_element_type_8, convert_element_type_9, div, exp, sub_1, sum_1

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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (16*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7 + tmp1
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 - tmp5
        tmp11 = tl.exp(tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr0 + (x0 + (16*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp15 + tmp1
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp5
        tmp19 = tl.exp(tmp18)
        tmp20 = tmp19 / tmp13
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (576*x1) + (331776*x0)), tmp21, rmask & xmask)


def get_args():
    arg_0 = rand_strided((576, 576, 16), (9216, 16, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16,), (1,), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((1, 16, 576, 576), (5308416, 331776, 576, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax_4.run(*args, 9216, 576, grid=grid(9216), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax_4.benchmark_all_configs(*args, 9216, 576, grid=grid(9216))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
