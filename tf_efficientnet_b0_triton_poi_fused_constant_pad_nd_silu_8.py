# kernel path: /data/liyang/pytorch/inductor_log/timm_models/tf_efficientnet_b0/amp_bf16/em/cemel5nmpisujp7wuwww6juiihziurnzl5syfrh5pigww3mklwmo.py
# Source Nodes: [getattr_getattr_l__self___blocks___1_____0___bn1_act, pad_1], Original ATen: [aten.constant_pad_nd, aten.silu]
# getattr_getattr_l__self___blocks___1_____0___bn1_act => convert_element_type_28, mul_16, sigmoid_4
# pad_1 => constant_pad_nd_1

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

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156905472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10848) % 113
    x1 = (xindex // 96) % 113
    x3 = (xindex // 1225824)
    x4 = xindex % 10848
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (10752*x2) + (1204224*x3)), tmp5, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.where(tmp5, tmp9, 0.0)
    tl.store(out_ptr0 + (x5), tmp10, None)


def get_args():
    arg_0 = rand_strided((128, 96, 112, 112), (1204224, 1, 10752, 96), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 96, 113, 113), (1225824, 1, 10848, 96), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_constant_pad_nd_silu_8.run(*args, 156905472, grid=grid(156905472), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_constant_pad_nd_silu_8.benchmark_all_configs(*args, 156905472, grid=grid(156905472))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
