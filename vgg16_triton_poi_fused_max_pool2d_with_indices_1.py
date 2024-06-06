# kernel path: /data/liyang/pytorch/inductor_log/torchbench/vgg16/amp_bf16/si/csioowavejvhpemixaedhi7smo2cdik6e4yc2tewvksaycz5eeuw.py
# Source Nodes: [l__self___features_4], Original ATen: [aten.max_pool2d_with_indices]
# l__self___features_4 => max_pool2d_with_indices

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

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 112
    x2 = (xindex // 7168)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (28672*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (28672*x2)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (128*x1) + (28672*x2)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + (128*x1) + (28672*x2)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)


def get_args():
    arg_0 = rand_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_max_pool2d_with_indices_1.run(*args, 3211264, grid=grid(3211264), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_max_pool2d_with_indices_1.benchmark_all_configs(*args, 3211264, grid=grid(3211264))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
