# kernel path: /data/liyang/pytorch/inductor_log/torchbench/hf_T5/amp_bf16/bw/cbwh7cox6g7yh6dgg6lcytcskznztsv4t2x3queoc2du6w5y27kz.py
# Source Nodes: [float_9, softmax_6, type_as_6], Original ATen: [aten._softmax, aten._to_copy]
# float_9 => convert_element_type_104
# softmax_6 => amax_6, div_10, exp_6, sub_11, sum_7
# type_as_6 => convert_element_type_105

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
    size_hints=[16384, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_3(in_ptr0, in_ptr1, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp34 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = (-1)*(tl.minimum(0, r2 + ((-1)*x0), tl.PropagateNan.NONE))
        tmp3 = tl.full([1, 1], 16, tl.int64)
        tmp4 = tmp2 < tmp3
        tmp5 = tmp2.to(tl.float32)
        tmp6 = 16.0
        tmp7 = tmp5 / tmp6
        tmp8 = tl.log(tmp7)
        tmp9 = 2.0794415416798357
        tmp10 = tmp8 / tmp9
        tmp11 = tmp10 * tmp6
        tmp12 = tmp11.to(tl.int64)
        tmp13 = tmp12 + tmp3
        tmp14 = tl.full([1, 1], 31, tl.int64)
        tmp15 = triton_helpers.minimum(tmp13, tmp14)
        tmp16 = tl.where(tmp4, tmp2, tmp15)
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.where(tmp18 < 0, tmp18 + 32, tmp18)
        # tl.device_assert((0 <= tmp19) & (tmp19 < 32), "index out of bounds: 0 <= tmp19 < 32")
        tmp20 = tl.load(in_ptr1 + (x1 + (8*tmp19)), None, eviction_policy='evict_first')
        tmp21 = r2
        tmp22 = x0
        tmp23 = tmp21 <= tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp25 = 1.0
        tmp26 = tmp25 - tmp24
        tmp27 = -3.4028234663852886e+38
        tmp28 = tmp26 * tmp27
        tmp29 = tmp20 + tmp28
        tmp30 = tmp1 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = triton_helpers.maximum(_tmp34, tmp33)
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr0 + (r2 + (2048*x3)), tmp32, rmask)
    tmp34 = triton_helpers.max2(_tmp34, 1)[:, None]
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp36 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tmp36 - tmp34
        tmp38 = tl.exp(tmp37)
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp42 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tmp42 - tmp34
        tmp44 = tl.exp(tmp43)
        tmp45 = tmp44 / tmp40
        tmp46 = tmp45.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp46, rmask)


def get_args():
    arg_0 = rand_strided((8, 2048, 2048), (4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((32, 8), (8, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='xpu:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 8, 2048, 2048), (33554432, 4194304, 2048, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax__to_copy_3.run(*args, 16384, 2048, grid=grid(16384), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax__to_copy_3.benchmark_all_configs(*args, 16384, 2048, grid=grid(16384))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
