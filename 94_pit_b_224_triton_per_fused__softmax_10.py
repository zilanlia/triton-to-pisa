
import triton
import triton.language as tl

import triton_helpers
import intel_extension_for_pytorch 

from helper import rand_strided, size
import torch

import argparse
import sys

parser = argparse.ArgumentParser(description='function parameters')
parser.add_argument('--device', dest='device', type=str, help='Running on hw or sim')
parser.add_argument('--shape0', dest='shape0', type=tuple, help='shape of arg0')
parser.add_argument('--shape1', dest='shape1', type=tuple, help='shape of arg1')

@triton.jit
def triton_per_fused__softmax_10(in_ptr0, out_ptr2, xnumel, rnumel, arg0_x):
    XBLOCK: tl.constexpr = 1
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (arg0_x * x0)), rmask & xmask, other=0).to(tl.float32)
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
    tl.store(out_ptr2 + (r1 + (arg0_x * x0)), tmp13, rmask & xmask)


def get_args(arg_0_size, arg_1_size):
    # WR  as_strided will cause error "Connection Closed"
    # arg_0 = rand_strided(arg_0_shape, arg_0_stride, device='xpu:0', dtype=torch.bfloat16)
    # arg_1 = rand_strided(arg_1_shape, arg_1_stride, device='xpu:0', dtype=torch.bfloat16)
    arg_0 = torch.rand(arg_0_size, device='xpu:0', dtype=torch.bfloat16)
    arg_1 = torch.rand(arg_1_size, device='xpu:0', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(arg_0, arg_1, arg_0_shape, arg_0_stride, arg_0_size):
    grid=lambda meta: (triton.cdiv(arg_0_size, 1024), )
    triton_per_fused__softmax_10[grid](arg_0, arg_1, arg_0_size, arg_0_stride[1], arg_0_shape[0])

if __name__ == '__main__':
    args = parser.parse_args()
    arg_0_shape = ()
    arg_1_shape = ()

    if args.shape0 and  args.shape1:
        arg_0_shape = arg_0_shape + args.shape0
        arg_1_shape = arg_1_shape + args.shape1
    elif args.device == "hw":
        arg_0_shape = arg_0_shape + (256, 962, 962)
        arg_1_shape = arg_1_shape + (64, 4, 962, 962)
    elif args.device == "sim":
        arg_0_shape = arg_0_shape + (1, 4, 1024)
        arg_1_shape = arg_1_shape + (1, 1, 4, 1024)
    else:
        print("need device info or args' shape")
        sys.exit(0)

    if (not len(arg_0_shape) == 3) :
        print("shape of arg0 must be 3")
    if (not len(arg_1_shape) == 4) :
        print("shape of arg1 must be 4")

    print(arg_0_shape)
    print(arg_1_shape)
    print(len(arg_0_shape))

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
    arg_0, arg_1 = get_args(arg_0_size, arg_1_size)

    print("call func")
    call(arg_0, arg_1, arg_0_shape, arg_0_stride, arg_0_size)

    print("result")
    print(arg_1)