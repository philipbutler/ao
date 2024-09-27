# Simplest test cases for comparison

import torch
from torch import tensor, Tensor

# A few steps of standard AdamW FP32

a = tensor([0.8477, 0.3092, 0.2363, 0.2300], device='cuda')        # can use torch.rand(4)
a.grad = tensor([0.8530, 0.7153, 0.1018, 0.4003], device='cuda')   # fake gradient via torch.rand(4)
o = torch.optim.AdamW([a], fused=True)                             # AdamW Optimizer

for i in range(3):
    o.step()
    print("Step " + str(i) + ": " + str(a))

'''
Step 0: a == tensor([0.8467, 0.3082, 0.2353, 0.2290])
Step 1: a == tensor([0.8457, 0.3072, 0.2343, 0.2280])
Step 2: a == tensor([0.8447, 0.3062, 0.2333, 0.2270])

              ^^^ Notice these are different lol ^^^
'''

# Up next: match TorchAO's FP8 quantization

# `python setup.py develop` is giving me errors (ssh-ing Lambdalabs)
# so just copy-pasting instead of importing

# from torchao/prototype/low_bit_optim/subclass_fp8.pygi
# https://github.com/pytorch/ao/blob/0bdde92114b470823aa24725bf3b0811e980c8ce/torchao/prototype/low_bit_optim/subclass_fp8.py#L13C1-L19C36

DTYPE = torch.float8_e4m3fn

def quantize_fp8(input: Tensor, block_size: int):
    shape = input.shape
    input = input.view(-1, block_size)                                      # will start with just shape [32]
    scale = input.abs().amax(-1).clip(1e-12) / torch.finfo(DTYPE).max       # absolute value, max across the last dimension (0th here), clip(num) == clip(min=num), max for (DTYPE = torch.float8_e4m3fn) is 448
    input = input / scale.view(-1, 1)                                       # flatten
    codes = input.to(DTYPE).view(-1)                                        # convert to torch.float8_e4m3fn, squeeze
    return codes.view(shape), scale                                         # (original shape, scale) [scale for dquntzing]

                                                                            # noticing there's no zero-point

# 32 for unit testing, but 2048 cause Arun & Erik are the source of truth
block_size = 32

# a = torch.rand(block_size, device='cuda')
a = tensor([0.08156189322471619, 0.3785102963447571, 0.23286126554012299, 0.9647358655929565, 0.4282546639442444, 0.7482216954231262, 0.903114378452301, 0.3822559118270874, 0.3563106954097748, 0.39088377356529236, 0.2661018669605255, 0.45732927322387695, 0.356448769569397, 0.5366447567939758, 0.9373241662979126, 0.2961907982826233, 0.8248701095581055, 0.6990491151809692, 0.002520027570426464, 0.9591174125671387, 0.9756536483764648, 0.493215948343277, 0.678508996963501, 0.8220535516738892, 0.3433856666088104, 0.012765476480126381, 0.9194097518920898, 0.7243597507476807, 0.30336636304855347, 0.8506981134414673, 0.9834323525428772, 0.3326418697834015], device='cuda')

# fake_grad
# g = torch.rand(block_size, device='cuda')
g = tensor([0.7289007306098938, 0.30462440848350525, 0.9082905054092407, 0.31704971194267273, 0.19741280376911163, 0.5811731815338135, 0.9425305724143982, 0.43781372904777527, 0.09683270007371902, 0.12920717895030975, 0.8269669413566589, 0.7294973134994507, 0.9390449523925781, 0.155783012509346, 0.5775147676467896, 0.6951613426208496, 0.49144434928894043, 0.16329661011695862, 0.2072339653968811, 0.27448904514312744, 0.43389183282852173, 0.8969299793243408, 0.6707720160484314, 0.3562951683998108, 0.9982314109802246, 0.4646815061569214, 0.560585081577301, 0.9774811863899231, 0.6622148752212524, 0.19557878375053406, 0.23262782394886017, 0.802483081817627], device="cuda")

#print('params', a)
#print('grads', g)

#quantize_fp8(a, block_size)

print("\nverbose params:", [float(i) for i in a[:]])
print("\nverbose grads:", [float(i) for i in g[:]])

s = str(['fp8(' + str(float(i)) + ')' for i in a]).replace("'", "")
print("\nstring for cuda code params:", s)

s = str(['fp8(' + str(float(i)) + ')' for i in g]).replace("'", "")
print("\nstring for cuda code grads:", s)
