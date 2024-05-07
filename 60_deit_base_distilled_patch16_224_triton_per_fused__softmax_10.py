
import triton
import triton.language as tl

import torch
import intel_extension_for_pytorch 

from helper import rand_strided, size
import triton_helpers

import argparse
import sys

parser = argparse.ArgumentParser(description='function parameters')
parser.add_argument('--device', dest='device', type=str, help='Running on hw or sim')
parser.add_argument('--shape0', dest='shape0', type=tuple, help='shape of arg0')
parser.add_argument('--shape1', dest='shape1', type=tuple, help='shape of arg1')

@triton.jit
def triton_per_fused__softmax_10(in_ptr0, out_ptr2, xnumel, rnumel, arg0_z, XBLOCK : tl.constexpr):
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (arg0_z*x0)), rmask & xmask, other=0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, float("-inf"))
    tmp5 = triton_helpers.max2(tmp4, 1)[:, None]
    tmp6 = tmp1 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tmp7 / tmp11
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (arg0_z*x0)), tmp13, rmask & xmask)


def get_args(arg_0_size, arg_1_size):
    # WR  as_strided will cause error "Connection Closed"
    # arg_0 = rand_strided((768, 198, 198), (39204, 198, 1), device='xpu:0', dtype=torch.bfloat16)
    # arg_1 = rand_strided((64, 12, 198, 198), (470448, 39204, 198, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_0 = torch.rand(arg_0_size, device='xpu:0', dtype=torch.bfloat16)
    arg_1 = torch.rand(arg_1_size, device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args, arg_0_shape):
    thread_num = arg_0_shape[0] * arg_0_shape[1]
    grid=lambda meta: (triton.cdiv(thread_num, 128), )
    print(grid)
    triton_per_fused__softmax_10[grid](*args, thread_num, arg_0_shape[2], arg_0_shape[2], 128)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_0_shape = ()
    arg_1_shape = ()

    if args.shape0 and  args.shape1:
        arg_0_shape = arg_0_shape + args.shape0
        arg_1_shape = arg_1_shape + args.shape1
    elif args.device == "hw":
        arg_0_shape = arg_0_shape + (768, 198, 198)
        arg_1_shape = arg_1_shape + (64, 12, 198, 198)
    elif args.device == "sim":
        arg_0_shape = arg_0_shape + (4, 4, 128)
        arg_1_shape = arg_1_shape + (2, 2, 4, 128)
    else:
        print("need device info or args' shape")
        sys.exit(0)

    if (not len(arg_0_shape) == 3) :
        print("shape of arg0 must be 3")
    if (not len(arg_1_shape) == 4) :
        print("shape of arg1 must be 4")

    arg_0_stride = (arg_0_shape[1] * arg_0_shape[2], 
                arg_0_shape[2], 
                1)
    arg_1_stride = (arg_1_shape[1] * arg_1_shape[2] * arg_1_shape[3],
                arg_1_shape[2] * arg_1_shape[3],
                arg_1_shape[3], 
                1)

    arg_0_size = size(arg_0_shape)
    arg_1_size = size(arg_1_shape)

    print("Creare input data")
    args = get_args(arg_0_size, arg_1_size)

    print("call func")
    call(args, arg_0_shape)

    print("result")
    print(args[1])