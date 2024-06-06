# kernel path: /data/liyang/pytorch/inductor_log/huggingface/OPTForCausalLM/amp_bf16/xj/cxja3ea7lo5scz2hkeradlvvbectprrpqdf7j3w2q2rsfj4bvmid.py
# Source Nodes: [bmm_1, softmax], Original ATen: [aten._softmax, aten._to_copy]
# bmm_1 => convert_element_type_12
# softmax => amax, div, exp, sub_3, sum_1

import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

from torch._dynamo.testing import rand_strided
import torch
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
from torch._inductor.triton_heuristics import grid

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_3(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp13 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = r2
        tmp3 = 1 + x0
        tmp4 = tmp2 < tmp3
        tmp5 = 0.0
        tmp6 = -3.4028234663852886e+38
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tl.full([1, 1], False, tl.int1)
        tmp9 = tl.where(tmp8, tmp6, tmp7)
        tmp10 = tmp1 + tmp9
        tmp11 = triton_helpers.maximum(tmp10, tmp6)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = triton_helpers.maximum(_tmp13, tmp12)
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = triton_helpers.max2(_tmp13, 1)[:, None]
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = r2
        tmp18 = 1 + x0
        tmp19 = tmp17 < tmp18
        tmp20 = 0.0
        tmp21 = -3.4028234663852886e+38
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full([1, 1], False, tl.int1)
        tmp24 = tl.where(tmp23, tmp21, tmp22)
        tmp25 = tmp16 + tmp24
        tmp26 = triton_helpers.maximum(tmp25, tmp21)
        tmp27 = tmp26 - tmp13
        tmp28 = tl.exp(tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = r2
        tmp35 = 1 + x0
        tmp36 = tmp34 < tmp35
        tmp37 = 0.0
        tmp38 = -3.4028234663852886e+38
        tmp39 = tl.where(tmp36, tmp37, tmp38)
        tmp40 = tl.full([1, 1], False, tl.int1)
        tmp41 = tl.where(tmp40, tmp38, tmp39)
        tmp42 = tmp33 + tmp41
        tmp43 = triton_helpers.maximum(tmp42, tmp38)
        tmp44 = tmp43 - tmp13
        tmp45 = tl.exp(tmp44)
        tmp46 = tmp45 / tmp30
        tmp47 = tmp46.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp47, rmask)


def get_args():
    arg_0 = rand_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax__to_copy_3.run(*args, 49152, 2048, grid=grid(49152), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax__to_copy_3.benchmark_all_configs(*args, 49152, 2048, grid=grid(49152))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
