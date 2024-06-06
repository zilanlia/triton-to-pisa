# kernel path: /data/liyang/pytorch/inductor_log/torchbench/mobilenet_v3_large/amp_bf16/tr/ctrqst5n536ddkivquejhhnutj5zpsbqf73x3zyxuughpydqlr2t.py
# Source Nodes: [getattr_l__self___features___14___block_1_2], Original ATen: [aten.hardswish]
# getattr_l__self___features___14___block_1_2 => add_112, clamp_max_22, clamp_min_22, convert_element_type_234, div_22, mul_145

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

@pointwise(size_hints=[32768, 64], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_hardswish_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (960*x2) + (47040*y1)), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp9 = tmp8.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp9, xmask)


def get_args():
    arg_0 = rand_strided((32, 960, 7, 7), (47040, 1, 6720, 960), device='xpu:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 960, 7, 7), (47040, 49, 7, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_hardswish_21.run(*args, 30720, 49, grid=grid(30720, 49), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_hardswish_21.benchmark_all_configs(*args, 30720, 49, grid=grid(30720, 49))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
