# kernel path: /data/liyang/pytorch/inductor_log/huggingface/BlenderbotSmallForConditionalGeneration/amp_bf16/sz/cszfoebsjx2ycoi6vqrxdcyxsqtin6pbpdit33bjgy4oa36hga4s.py
# Source Nodes: [add_51, cross_entropy], Original ATen: [aten._log_softmax, aten.add]
# add_51 => add_151
# cross_entropy => amax_24, exp_24, sub_66, sum_25

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
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_add_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__log_softmax_add_11(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 50265
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50265*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (r1 + (50265*x0)), tmp3, rmask)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp5, None)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(out_ptr0 + (r1 + (50265*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 - tmp5
        tmp9 = tl.exp(tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp11, None)


def get_args():
    arg_0 = rand_strided((8192, 50265), (50265, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 50265), (50265, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 128, 50265), (6433920, 50265, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((8192, 1), (1, 8192), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((8192, 1), (1, 8192), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__log_softmax_add_11.run(*args, 8192, 50265, grid=grid(8192), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__log_softmax_add_11.benchmark_all_configs(*args, 8192, 50265, grid=grid(8192))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
