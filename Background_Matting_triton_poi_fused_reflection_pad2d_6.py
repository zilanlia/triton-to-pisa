# kernel path: /data/liyang/pytorch/inductor_log/torchbench/Background_Matting/amp_bf16/3b/c3btdqsrdwe7o5hc6xukltsvv3557tt56pmdzuxfvkqlo6cvhbkx.py
# Source Nodes: [getattr_l__self___model_res_dec___3___conv_block_0], Original ATen: [aten.reflection_pad2d]
# getattr_l__self___model_res_dec___3___conv_block_0 => reflection_pad2d_4

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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_reflection_pad2d_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4326400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 33280)
    x1 = (xindex // 256) % 130
    x0 = xindex % 256
    x5 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.abs(tmp0)
    tmp2 = tl.full([1], 127, tl.int32)
    tmp3 = tmp2 - tmp1
    tmp4 = tl.abs(tmp3)
    tmp5 = tmp2 - tmp4
    tmp6 = (-1) + x1
    tmp7 = tl.abs(tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = tl.abs(tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (256*tmp10) + (32768*tmp5)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp11, xmask)


def get_args():
    arg_0 = rand_strided((1, 256, 128, 128), (4194304, 1, 32768, 256), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 256, 130, 130), (4326400, 1, 33280, 256), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_reflection_pad2d_6.run(*args, 4326400, grid=grid(4326400), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_reflection_pad2d_6.benchmark_all_configs(*args, 4326400, grid=grid(4326400))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
