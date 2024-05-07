
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
def triton_red_fused__log_softmax__to_copy_21(in_ptr0, out_ptr2, xnumel, rnumel, consta_val_0, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (rnumel*(x0 % 511)) + (consta_val_0*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (rnumel*(x0 % 511)) + (consta_val_0*(x0 // 511))), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (rnumel*(x0 % 511)) + (consta_val_0*(x0 // 511))), rmask & xmask, other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl.log(tmp10)
        tmp16 = tmp14 - tmp15
        tl.store(out_ptr2 + (r1 + (rnumel*x0)), tmp16, rmask & xmask)


def get_args(arg_0_size, arg_1_size):
    # WR  torch.as_strided will cause error "Connection Closed" in pre-si env
    # arg_0 = rand_strided((8192, 50257), (50257, 1), device='xpu:0', dtype=torch.bfloat16)
    # arg_1 = rand_strided((8176, 50257), (50257, 1), device='xpu:0', dtype=torch.float32)
    arg_0 = torch.rand(arg_0_size, device='xpu:0', dtype=torch.bfloat16)
    arg_1 = torch.rand(arg_1_size, device='xpu:0', dtype=torch.float32)
    return arg_0, arg_1,


def call(args, arg_1_shape):
    grid=lambda meta: (triton.cdiv(arg_1_shape[0], 1024), )
    consta_val_0 = 512 * arg_1_shape[1]
    triton_red_fused__log_softmax__to_copy_21[grid](*args, arg_1_shape[0], arg_1_shape[1], consta_val_0, 1, 1024)


if __name__ == '__main__':
    args = parser.parse_args()
    arg_0_shape = ()
    arg_1_shape = ()

    if args.shape0 and  args.shape1:
        arg_0_shape = arg_0_shape + args.shape0
        arg_1_shape = arg_1_shape + args.shape1
    elif args.device == "hw":
        arg_0_shape = arg_0_shape + (8192, 50257)
        arg_1_shape = arg_1_shape + (8176, 50257)
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
    call(args, arg_1_shape)

    print("result")
    print(args[1])