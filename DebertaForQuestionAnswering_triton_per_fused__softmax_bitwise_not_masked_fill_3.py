# kernel path: /data/liyang/pytorch/inductor_log/huggingface/DebertaForQuestionAnswering/amp_bf16/ww/cwwghg46l543yw7y6suujwzr62h7kviff4otz2zpqn4fcpfyq5gf.py
# Source Nodes: [invert, masked_fill, masked_fill_, softmax], Original ATen: [aten._softmax, aten.bitwise_not, aten.masked_fill]
# invert => full_default_2
# masked_fill => full_default_3, where
# masked_fill_ => full_default_4, where_1
# softmax => amax, convert_element_type_5, convert_element_type_6, div_2, exp, sub_2, sum_1

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
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_bitwise_not_masked_fill_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax_bitwise_not_masked_fill_3(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 98304
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.full([1], False, tl.int1)
    tmp2 = -3.3895313892515355e+38
    tmp3 = tl.where(tmp1, tmp2, tmp0)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tmp9 = tmp4 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp15 = tmp10 / tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 0.0
    tmp18 = tl.where(tmp1, tmp17, tmp16)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp18, rmask)


def get_args():
    arg_0 = rand_strided((192, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16, 12, 512, 512), (3145728, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax_bitwise_not_masked_fill_3.run(*args, 98304, 512, grid=grid(98304), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax_bitwise_not_masked_fill_3.benchmark_all_configs(*args, 98304, 512, grid=grid(98304))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
