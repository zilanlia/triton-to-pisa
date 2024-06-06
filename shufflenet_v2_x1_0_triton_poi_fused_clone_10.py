# kernel path: /data/liyang/pytorch/inductor_log/torchbench/shufflenet_v2_x1_0/amp_bf16/yr/cyrpwut7jnhnwqrrtofcjqdvlq6isidcnnlpb2dxonuyz6i73wra.py
# Source Nodes: [contiguous_4], Original ATen: [aten.clone]
# contiguous_4 => clone_4

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

@pointwise(size_hints=[8192, 512], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7424
    xnumel = 392
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 196
    x3 = (xindex // 196)
    y0 = yindex % 116
    y1 = (yindex // 116)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (116*x3) + (232*x2) + (45472*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x5 + (392*y4)), tmp0, xmask & ymask)


def get_args():
    arg_0 = rand_strided((64, 232, 14, 14), (45472, 1, 3248, 232), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 116, 2, 14, 14), (45472, 392, 196, 14, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_clone_10.run(*args, 7424, 392, grid=grid(7424, 392), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_clone_10.benchmark_all_configs(*args, 7424, 392, grid=grid(7424, 392))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
