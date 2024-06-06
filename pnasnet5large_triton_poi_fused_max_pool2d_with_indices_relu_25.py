# kernel path: /data/liyang/pytorch/inductor_log/timm_models/pnasnet5large/amp_bf16/2d/c2dblbtqs44jplyl65h2aju4bavl343p7nw7huipzho2a5nd3hha.py
# Source Nodes: [l__self___cell_0_comb_iter_0_left_act_1, l__self___cell_0_comb_iter_0_right], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
# l__self___cell_0_comb_iter_0_left_act_1 => relu_31
# l__self___cell_0_comb_iter_0_right => max_pool2d_with_indices_6

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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_relu_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_relu_25(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 42
    x1 = (xindex // 216) % 42
    x6 = xindex
    tmp61 = tl.load(in_ptr0 + (x6), xmask).to(tl.float32)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9288) + x6), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-9072) + x6), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp12)
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-8856) + x6), tmp25 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp20)
    tmp29 = x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-216) + x6), tmp33 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = triton_helpers.maximum(tmp35, tmp28)
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (x6), tmp37 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp36)
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (216 + x6), tmp41 & xmask, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = triton_helpers.maximum(tmp43, tmp40)
    tmp45 = 1 + x2
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (8856 + x6), tmp49 & xmask, other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp44)
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (9072 + x6), tmp53 & xmask, other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (9288 + x6), tmp57 & xmask, other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tmp62 = triton_helpers.maximum(0, tmp61)
    tl.store(out_ptr0 + (x6), tmp60, xmask)
    tl.store(out_ptr1 + (x6), tmp62, xmask)


def get_args():
    arg_0 = rand_strided((16, 216, 42, 42), (381024, 1, 9072, 216), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((16, 216, 42, 42), (381024, 1, 9072, 216), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((16, 216, 42, 42), (381024, 1, 9072, 216), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_poi_fused_max_pool2d_with_indices_relu_25.run(*args, 6096384, grid=grid(6096384), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused_max_pool2d_with_indices_relu_25.benchmark_all_configs(*args, 6096384, grid=grid(6096384))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
