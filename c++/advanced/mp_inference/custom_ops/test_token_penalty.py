import paddle
import numpy as np
from custom_setup_ops import get_token_penalty, get_token_penalty_only_once

paddle.seed(2023)

pre_ids = paddle.randint(0, 10000, (8, 1000))
pre_ids[:, -1] = pre_ids[:, -2]
print(pre_ids)
logits = paddle.rand(shape=[8, 10000], dtype='float16')
penalty_scores = np.array([1.2] * 8).astype(np.float16).reshape(-1, 1)
penalty_scores = paddle.to_tensor(penalty_scores)
# penalty_scores = paddle.rand(shape=[8, 1])

print("logits[0][pre_ids[0]]: ", logits[0][pre_ids[0]])
# res = get_token_penalty(pre_ids, logits, penalty_scores)
res = get_token_penalty_only_once(pre_ids, logits, penalty_scores)
for i in range(8):
    print("res[{}]:{}".format(i, res[i][pre_ids[i]]))


input_ids = pre_ids
score = paddle.index_sample(logits, input_ids)
score = paddle.where(score < 0, score * penalty_scores, score / penalty_scores)

bsz = paddle.shape(logits)[0] # TODO: Bsz as input for inference with dynamic batch_size
bsz_range = paddle.arange(start=bsz*0, end=bsz, step=bsz/bsz, name='bsz_range').unsqueeze(-1)
input_ids = input_ids + bsz_range * logits.shape[-1]
res2 = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(logits.shape) 
print("-------------------------------------------")
for i in range(8):
    print(res2[i][pre_ids[i]])

print("res_sub:")
for i in range(8):
    print(res2[i][pre_ids[i]] - res[i][pre_ids[i]])

print((res.numpy() - res2.numpy()).sum())