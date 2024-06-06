# kernel path: /data/liyang/pytorch/inductor_log/timm_models/gmlp_s16_224/amp_bf16/zk/czkvvuauqldapma433rk5a5ybc3fcgcuse53wymmp3pms5zu2etd.py
# Source Nodes: [mul], Original ATen: [aten.mul]
# mul => mul_7

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

@pointwise(size_hints=[32768, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
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
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 * tmp13
    tl.store(out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)


def get_args():
    arg_0 = rand_strided((25088, 1536), (1536, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 768, 196), (150528, 196, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((196,), (1,), device='xpu:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((128, 196, 768), (150528, 768, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_mul_3.run(*args, 25088, 768, grid=grid(25088, 768), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_mul_3.benchmark_all_configs(*args, 25088, 768, grid=grid(25088, 768))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
