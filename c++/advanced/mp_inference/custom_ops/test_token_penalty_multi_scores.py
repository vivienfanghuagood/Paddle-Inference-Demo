import paddle
import numpy as np
from custom_setup_ops import get_token_penalty_multi_scores, get_token_penalty_only_once

paddle.seed(2023)

penalty_score = 1.2
frequency_score = 0.8
presence_score = 0.6
bs = 8

type_now = "float32"
penalty_score_tensor = paddle.to_tensor([penalty_score,] * 8, type_now).reshape(-1, 1)
frequency_score_tensor = paddle.to_tensor([frequency_score,] * 8, type_now).reshape(-1, 1)
presence_score_tensor = paddle.to_tensor([presence_score, ] * 8, type_now).reshape(-1, 1)

pre_ids = paddle.randint(0, 10000, (8, 1000))
print(pre_ids)
logits = paddle.rand(shape=[8, 10000], dtype='float16')
penalty_scores = np.array([penalty_score] * 8).astype(np.float16).reshape(-1, 1)
penalty_scores = paddle.to_tensor(penalty_scores)

res = get_token_penalty_only_once(pre_ids, logits, penalty_scores)

res2 = get_token_penalty_multi_scores(pre_ids, logits, penalty_score_tensor, frequency_score_tensor, presence_score_tensor)
