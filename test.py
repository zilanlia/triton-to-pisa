
import triton
import triton.language as tl
# from torch._inductor.ir import ReductionHint
# from torch._inductor.ir import TileHint
# from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
# from torch._inductor.utils import instance_descriptor
import triton_helpers
import intel_extension_for_pytorch 

from helper import rand_strided
import torch
# from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream
# from torch._inductor.triton_heuristics import grid

# @persistent_reduction(
#     size_hints=[262144, 1024],
#     reduction_hint=ReductionHint.INNER,
#     filename=__file__,
#     meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
# )

#abstract parameters
# arg_0_shape = (1, 4, 1024)
# arg_0_stride = (arg_0_shape[1] * arg_0_shape[2], 
#                 arg_0_shape[2], 
#                 1)
# arg_1_shape = (1, 1, 4, 1024)
# arg_1_stride = (arg_1_shape[1] * arg_1_shape[2] * arg_1_shape[3],
#                 arg_1_shape[2] * arg_1_shape[3],
#                 arg_1_shape[3], 
#                 1)


@triton.jit
def triton_per_fused__softmax_10(in_ptr0, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp4, 0))
    tmp6 = tmp1 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp7 / tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp13, rmask & xmask)


def get_args():
    arg_0 = rand_strided((1, 4, 1024), (4096, 1024, 1), device='xpu', dtype=torch.bfloat16)
    arg_1 = rand_strided((1, 1, 4, 1024), (4096, 4096, 1024, 4), device='xpu', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args):
    grid=lambda meta: (triton.cdiv(4096, 1024), )
    triton_per_fused__softmax_10[grid](*args, 4096, 1024)


if __name__ == '__main__':
    print("Creare input data")
    args = get_args()
    print("call func")
    call(args)
    print("result")
    arg0,arg1 = *args
    print(arg1)