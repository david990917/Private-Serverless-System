# 只是为了测试 import torch 的时间

import time

# 第一次 import torch
time0 = time.time()
import torch

print("第一次 import torch", time.time() - time0)


time0 = time.time()
import torch
print("第二次 import torch", time.time() - time0)

time0 = time.time()
import torch
print("第三次 import torch", time.time() - time0)

time0 = time.time()
import torch
print("第四次 import torch", time.time() - time0)