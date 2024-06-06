# kernel path: /data/liyang/pytorch/inductor_log/timm_models/volo_d1_224/amp_bf16/ix/cix2pdgaadltru7ibu2xnkrsikll63xrxkfuexui6rt65siajbug.py
# Source Nodes: [fold], Original ATen: [aten.col2im]
# fold => _unsafe_index_put, full_default

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

@pointwise(size_hints=[16384, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_col2im_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 3
    x3 = (xindex // 3) % 14
    x4 = (xindex // 42) % 3
    x5 = (xindex // 126)
    y0 = yindex % 192
    y1 = (yindex // 192)
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + ((32*x2) + (96*x4) + (288*x3) + (4032*x5) + (56448*((x2 + (3*x4) + (9*y0)) // 288)) + (338688*y1) + (((x2 + (3*x4) + (9*y0)) // 9) % 32)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.atomic_add(out_ptr0 + (x2 + (2*x3) + (30*x4) + (60*x5) + (900*y6)), tmp1, xmask)


def get_args():
    arg_0 = rand_strided((75264, 9, 32), (288, 32, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 192, 30, 30), (172800, 900, 30, 1), device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_col2im_7.run(*args, 12288, 1764, grid=grid(12288, 1764), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_col2im_7.benchmark_all_configs(*args, 12288, 1764, grid=grid(12288, 1764))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
