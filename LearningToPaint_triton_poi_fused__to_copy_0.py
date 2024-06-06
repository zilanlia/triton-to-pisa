# f1bb516d94b85adb95a40bb82cb6720fc92d190d3b0a47ebfa85eb88c5f9aec5
# kernel path: /data/liyang/pytorch/inductor_log/torchbench/LearningToPaint/amp_bf16/ih/cihicqjww2qiv33a2cfbbr7vlqhxhejyviefhif5ikvmpia4crdi.py
# Source Nodes: [l__self___conv1], Original ATen: [aten._to_copy]
# l__self___conv1 => convert_element_type

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

@pointwise(size_hints=[1024, 16384], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 9
    y1 = (yindex // 9)
    tmp0 = tl.load(in_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (y0 + (9*x2) + (147456*y1)), tmp1, ymask)


def get_args():
    arg_0 = rand_strided((96, 9, 128, 128), (147456, 16384, 128, 1), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((96, 9, 128, 128), (147456, 1, 1152, 9), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused__to_copy_0.run(*args, 864, 16384, grid=grid(864, 16384), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__to_copy_0.benchmark_all_configs(*args, 864, 16384, grid=grid(864, 16384))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
