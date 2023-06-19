import paddle
from custom_setup_ops import set_stop_value

topk_ids = paddle.randint(0, 10000, (8, 1))
res = set_stop_value(topk_ids, 29980)
print(res)