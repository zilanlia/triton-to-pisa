# kernel path: /data/liyang/pytorch/inductor_log/timm_models/coat_lite_mini/amp_bf16/ny/cnybnldwi3xnvtoeyxtdyztoel4b2h2macx4wqr4qvdiqg6vczp3.py
# Source Nodes: [l__self___serial_blocks1_0_mlp_act], Original ATen: [aten.gelu]
# l__self___serial_blocks1_0_mlp_act => add_9, convert_element_type_25, convert_element_type_26, erf, mul_10, mul_8, mul_9

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

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_gelu_19(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205586432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp10, None)


def get_args():
    arg_0 = rand_strided((128, 3137, 512), (1606144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_gelu_19.run(*args, 205586432, grid=grid(205586432), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_gelu_19.benchmark_all_configs(*args, 205586432, grid=grid(205586432))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
