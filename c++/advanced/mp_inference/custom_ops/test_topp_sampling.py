import paddle
import numpy as np
from paddle.utils.cpp_extension import load
from custom_setup_ops import topp_sampling

paddle.seed(2022)

x = paddle.randn([4, 100000], dtype="float16")
x = paddle.nn.functional.softmax(x)
top_ps = paddle.to_tensor(np.array([0.9,] * 4).astype(np.float16))
print(x)
print(top_ps)
out = topp_sampling(x, top_ps)
print(out)