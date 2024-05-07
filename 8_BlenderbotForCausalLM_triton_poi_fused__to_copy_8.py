
import triton
import triton.language as tl

import torch
import intel_extension_for_pytorch

from helper import rand_strided, size

import argparse
import sys

parser = argparse.ArgumentParser(description='function parameters')
parser.add_argument('--device', dest='device', type=str, help='Running on hw or sim')
parser.add_argument('--shape0', dest='shape0', type=tuple, help='shape of arg0')
parser.add_argument('--shape1', dest='shape1', type=tuple, help='shape of arg1')

@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)


def get_args(arg_0_size, arg_1_size):
    # WR  torch.as_strided will cause error "Connection Closed" in pre-si env
    # arg_0 = rand_strided((10240, 2560), (2560, 1), device='xpu:0', dtype=torch.float32)
    # arg_1 = rand_strided((10240, 2560), (2560, 1), device='xpu:0', dtype=torch.bfloat16)
    arg_0 = torch.rand(arg_0_size, device='xpu', dtype=torch.float32)
    arg_1 = torch.rand(arg_1_size, device='xpu', dtype=torch.bfloat16)
    return arg_0, arg_1,


def call(args, arg_0_size):
    grid=lambda meta: (triton.cdiv(arg_0_size, 1024), ) #BLOCK_SIZE=1024
    triton_poi_fused__to_copy_8[grid](*args, arg_0_size, 1024)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_0_shape = ()
    arg_1_shape = ()

    if args.shape0 and  args.shape1:
        arg_0_shape = arg_0_shape + args.shape0
        arg_1_shape = arg_1_shape + args.shape1
    elif args.device == "hw":
        arg_0_shape = arg_0_shape + (10240, 2560)
        arg_1_shape = arg_1_shape + (10240, 2560)
    elif args.device == "sim":
        arg_0_shape = arg_0_shape + (16, 256)
        arg_1_shape = arg_1_shape + (16, 256)
    else:
        print("need device info or args' shape")
        sys.exit(0)

    if (not len(arg_0_shape) == 2) :
        print("shape of arg0 must be 2")
    if (not len(arg_1_shape) == 2) :
        print("shape of arg1 must be 2")

    arg_0_size = size(arg_0_shape)
    arg_1_size = size(arg_1_shape)

    print("create args")
    args = get_args(arg_0_size, arg_1_size)
    print("call kernel")
    call(args, arg_0_size)
    print("result")
    print(args[1])