# kernel path: /data/liyang/pytorch/inductor_log/timm_models/levit_128/amp_bf16/nr/cnrkv7idrcb47qsxnpyowzaym6gmmrczorcsjf5lwbbzglbqihar.py
# Source Nodes: [add, matmul_1, mul, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.add, aten.mul]
# add => add_13
# matmul_1 => convert_element_type_27
# mul => mul_18
# softmax => amax, div_3, exp, sub_5, sum_1

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
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_mul_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_mul_7(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 784
    tmp0 = tl.load(in_ptr0 + (r2 + (196*x3)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r2 + (196*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, float("-inf"))
    tmp9 = triton_helpers.max2(tmp8, 1)[:, None]
    tmp10 = tmp5 - tmp9
    tmp11 = tl.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tmp11 / tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (196*x3)), tmp17, rmask)


def get_args():
    arg_0 = rand_strided((512, 196, 196), (38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((4, 196, 196), (38416, 196, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 4, 196, 196), (153664, 38416, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax__to_copy_add_mul_7.run(*args, 100352, 196, grid=grid(100352), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax__to_copy_add_mul_7.benchmark_all_configs(*args, 100352, 196, grid=grid(100352))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
