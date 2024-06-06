# kernel path: /data/liyang/pytorch/inductor_log/huggingface/AllenaiLongformerBase/amp_bf16/4z/c4z7p4ahzlcjbtxerammt5pu4dlbdoaw4hzgxtpjh3fhhjggi4na.py
# Source Nodes: [pad_3], Original ATen: [aten.constant_pad_nd]
# pad_3 => constant_pad_nd_3

import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*i1', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37847040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x1 = (xindex // 770) % 48
    x2 = (xindex // 36960)
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (1024*(x1 // 12))), tmp2, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + (513*(x1 % 12)) + (6156*x2) + (6303744*((((12*(x1 // 12)) + (x1 % 12)) // 12) % 4))), tmp2, other=0.0).to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.load(in_ptr2 + ((12*x2) + (12288*(x1 // 12)) + (x1 % 12)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.load(in_ptr3 + ((12*x2) + (12288*(x1 // 12)) + (x1 % 12)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp3, tmp11, tmp10)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.where(tmp2, tmp13, 0.0)
    tl.store(out_ptr0 + (x0 + (770*x2) + (788480*x1)), tmp14, None)


def get_args():
    arg_0 = rand_strided((4, 1024), (1024, 1), device='xpu:0', dtype=torch.bool)
    arg_1 = rand_strided((4, 1024, 12, 513), (6303744, 6156, 513, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((4, 1024, 12, 1), (12288, 12, 1, 49152), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((4, 1024, 12, 1), (12288, 12, 1, 49152), device='xpu:0', dtype=torch.float32)
    arg_4 = rand_strided((48, 4, 256, 770), (788480, 197120, 770, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_constant_pad_nd_19.run(*args, 37847040, grid=grid(37847040), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_constant_pad_nd_19.benchmark_all_configs(*args, 37847040, grid=grid(37847040))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
