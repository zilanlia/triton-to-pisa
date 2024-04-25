
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
#     size_hints=[65536, 1024],
#     reduction_hint=ReductionHint.INNER,
#     filename=__file__,
#     meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]}
# )
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_10(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 65536
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tmp3 - tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp10, rmask)


def get_args():
    arg_0 = rand_strided((64, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((64, 1024, 1024), (1048576, 1024, 1), device='xpu:0', dtype=torch.float32)
    arg_2 = rand_strided((4, 16, 1024, 1024), (16777216, 1048576, 1024, 1), device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2,


def call(args):
    # with torch.xpu._DeviceGuard(0):
    #     torch.xpu.set_device(0)
    #     stream0 = get_xpu_stream(0)
    grid=lambda meta: (65536, )
    triton_per_fused__softmax_backward_data__to_copy_10[grid](*args, 65536, 1024)


# def benchmark_all_configs(args):
#     with torch.xpu._DeviceGuard(0):
#         torch.xpu.set_device(0)
#         return triton_per_fused__softmax_backward_data__to_copy_10.benchmark_all_configs(*args, 65536, 1024, grid=grid(65536))


if __name__ == '__main__':
    # from torch._inductor.utils import get_num_bytes
    # from intel_extension_for_pytorch._inductor.xpu.utils import do_bench

    args = get_args()
    call(args)
    # ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    # num_gb = get_num_bytes(*args, num_in_out_args=0) / 1e9
    # gb_per_s = num_gb / (ms / 1e3)
    # print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")