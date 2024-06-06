# kernel path: /data/liyang/pytorch/inductor_log/timm_models/tf_mixnet_l/amp_bf16/gr/cgrmlp7gnmwp2kpmlnbpe7vdiikocmj57vxdwfnl2utboioys5yf.py
# Source Nodes: [getattr_getattr_l__self___blocks___2_____1___bn1_act], Original ATen: [aten.silu]
# getattr_getattr_l__self___blocks___2_____1___bn1_act => convert_element_type_74, mul_43, sigmoid_4

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

@pointwise(size_hints=[65536, 1024], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_30', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_silu_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43008
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (336*x2) + (263424*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp3, xmask)


def get_args():
    arg_0 = rand_strided((128, 336, 28, 28), (263424, 1, 9408, 336), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 336, 28, 28), (263424, 784, 28, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_silu_30.run(*args, 43008, 784, grid=grid(43008, 784), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_silu_30.benchmark_all_configs(*args, 43008, 784, grid=grid(43008, 784))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
