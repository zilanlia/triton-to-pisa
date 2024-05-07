
import triton
import triton.language as tl

import torch
import intel_extension_for_pytorch

import triton_helpers
from helper import rand_strided, size

import argparse
import sys

parser = argparse.ArgumentParser(description='function parameters')
parser.add_argument('--device', dest='device', type=str, help='Running on hw or sim')
parser.add_argument('--shape0', dest='shape0', type=tuple, help='shape of arg0')
parser.add_argument('--shape1', dest='shape1', type=tuple, help='shape of arg1')

@triton.jit
def triton_poi_fused_relu_threshold_backward_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = triton_helpers.maximum(0, tmp0)
    tmp2 = 0.0
    tmp3 = tmp1 <= tmp2
    tl.store(out_ptr0 + (x0), tmp3, None)


def get_args(arg_0_size, arg_1_size):
    # WR  as_strided will cause error "Connection Closed"
    # arg_0 = rand_strided((16384, 512), (512, 1), device='xpu:0', dtype=torch.bfloat16)
    # arg_1 = rand_strided((128, 128, 512), (65536, 512, 1), device='xpu:0', dtype=torch.bool)
    arg_0 = torch.rand(arg_0_size, device='xpu:0', dtype=torch.bfloat16)
    arg_1 = torch.zeros(arg_1_size, device='xpu:0', dtype=torch.bool)
    return arg_0, arg_1,


def call(args, arg_0_size):
    grid=lambda meta: (triton.cdiv(arg_0_size, 1024), )
    triton_poi_fused_relu_threshold_backward_33[grid](*args, arg_0_size, 1)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_0_shape = ()
    arg_1_shape = ()

    if args.shape0 and  args.shape1:
        arg_0_shape = arg_0_shape + args.shape0
        arg_1_shape = arg_1_shape + args.shape1
    elif args.device == "hw":
        arg_0_shape = arg_0_shape + (16384, 512)
        arg_1_shape = arg_1_shape + (128, 128, 512)
    elif args.device == "sim":
        arg_0_shape = arg_0_shape + (16, 256)
        arg_1_shape = arg_1_shape + (4, 4, 256)
    else:
        print("need device info or args' shape")
        sys.exit(0)

    if (not len(arg_0_shape) == 2) :
        print("shape of arg0 must be 2")
    if (not len(arg_1_shape) == 3) :
        print("shape of arg1 must be 3")

    arg_0_stride = (arg_0_shape[1], 1)
    arg_1_stride = (arg_1_shape[1] * arg_1_shape[2] ,arg_1_shape[2], 1)

    arg_0_size = size(arg_0_shape)
    arg_1_size = size(arg_1_shape)

    print("Creare input data")
    args = get_args(arg_0_size, arg_1_size)

    print("call func")
    call(args, arg_0_size)

    print("result")
    print(args[1])
