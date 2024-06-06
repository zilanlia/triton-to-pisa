# kernel path: /data/liyang/pytorch/inductor_log/huggingface/XLNetLMHeadModel/amp_bf16/ib/cibqn3c5tkkxiesprovvyxxyixsctnw5c4tubznuxxvouhhre7hb.py
# Source Nodes: [add_2, add_3, index_select, mul, softmax], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul]
# add_2 => add_2
# add_3 => add_3
# index_select => index
# mul => mul_2
# softmax => amax, convert_element_type_13, convert_element_type_14, div_1, exp, sub, sum_1

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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_index_select_mul_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 16
    x2 = (xindex // 8192)
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.125
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = 0.0
        tmp15 = tmp13 + tmp14
        tmp16 = 0.125
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp24 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = 0.0
        tmp28 = tmp26 + tmp27
        tmp29 = 0.125
        tmp30 = tmp28 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 - tmp9
        tmp33 = tl.exp(tmp32)
        tmp34 = tmp33 / tmp22
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (512*x4)), tmp35, rmask)


def get_args():
    arg_0 = rand_strided((128, 512, 512), (262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((128, 512, 1024), (524288, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8, 16, 512, 512), (4194304, 262144, 512, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_xpu_stream(0)
        triton_red_fused__softmax_add_index_select_mul_4.run(*args, 65536, 512, grid=grid(65536), stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_red_fused__softmax_add_index_select_mul_4.benchmark_all_configs(*args, 65536, 512, grid=grid(65536))


if __name__ == '__main__':
    from torch._inductor.utils import get_num_bytes
    from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
