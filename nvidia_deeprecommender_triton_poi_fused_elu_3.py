# kernel path: /data/liyang/pytorch/inductor_log/torchbench/nvidia_deeprecommender/amp_bf16/ta/cta2dbxhkqa5ubnbf6n5lp72vrk5yswuz7zucaw5bvimsaefsmab.py
# Source Nodes: [selu_5], Original ATen: [aten.elu]
# selu_5 => convert_element_type_23, convert_element_type_24, expm1_5, gt_5, mul_15, mul_16, mul_17, where_5

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

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_elu_3(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50675456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp4 = 1.0507009873554805
    tmp5 = tmp1 * tmp4
    tmp6 = 1.0
    tmp7 = tmp1 * tmp6
    tmp8 = libdevice.expm1(tmp7)
    tmp9 = 1.7580993408473766
    tmp10 = tmp8 * tmp9
    tmp11 = tl.where(tmp3, tmp5, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)


def get_args():
    arg_0 = rand_strided((256, 197951), (197951, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_elu_3.run(*args, 50675456, grid=grid(50675456), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_elu_3.benchmark_all_configs(*args, 50675456, grid=grid(50675456))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=1) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
