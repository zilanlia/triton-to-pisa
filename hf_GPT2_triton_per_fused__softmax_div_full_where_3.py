# kernel path: /data/liyang/pytorch/inductor_log/torchbench/hf_GPT2/amp_bf16/mz/cmzryucvctcltr5ybs5dwgqs2gvsb5rvlekf776osco5uwfmdjxc.py
# Source Nodes: [full, full_1, softmax, truediv, where], Original ATen: [aten._softmax, aten.div, aten.full, aten.where]
# full => full_default
# full_1 => full_default_1
# softmax => amax, convert_element_type_3, convert_element_type_4, div_1, exp, sub_1, sum_1
# truediv => div
# where => where

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
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_full_where_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_div_full_where_3(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (r2 + (1024*x3)), rmask, other=0.0).to(tl.float32)
    tmp2 = 8.0
    tmp3 = tmp1 / tmp2
    tmp4 = -3.3895313892515355e+38
    tmp5 = tl.where(tmp0, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp9, 0))
    tmp11 = tmp6 - tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp12 / tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp18, rmask)


def get_args():
    arg_0 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((12, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax_div_full_where_3.run(*args, 12288, 1024, grid=grid(12288), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax_div_full_where_3.benchmark_all_configs(*args, 12288, 1024, grid=grid(12288))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
