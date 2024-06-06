# kernel path: /data/liyang/pytorch/inductor_log/huggingface/BartForCausalLM/amp_bf16/dg/cdgaknv3f5k4zvebbp6jxremzcenqsht6aw43eumpduzohyuorxb.py
# Source Nodes: [bmm_1, softmax], Original ATen: [aten._softmax, aten._to_copy]
# bmm_1 => convert_element_type_9
# softmax => amax, div, exp, sub_1, sum_1

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
    size_hints=[65536, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__softmax__to_copy_3(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 65536
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
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = r2
    tmp3 = 1 + x0
    tmp4 = tmp2 < tmp3
    tmp5 = 0.0
    tmp6 = -3.4028234663852886e+38
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp1 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp11, 0))
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = tmp14 / tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr2 + (r2 + (1024*x3)), tmp20, rmask)


def get_args():
    arg_0 = rand_strided((64, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_per_fused__softmax__to_copy_3.run(*args, 65536, 1024, grid=grid(65536), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_per_fused__softmax__to_copy_3.benchmark_all_configs(*args, 65536, 1024, grid=grid(65536))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
