#include "paddle/extension.h"


template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

template<typename T>
__global__ inline void min_length_logits_process(T* logits,
                                                 const int64_t *cur_len,
                                                 const int64_t *min_len,
                                                 const int64_t *eos_token_id,
                                                 const int64_t bs,
                                                 const int64_t length,
                                                 const int64_t end_length) {
    int bi = blockIdx.x;
    if (cur_len[bi] < 0) {
        return;
    }
    if (cur_len[bi] < min_len[bi]) {
        for (int i=0; i < end_length; i++) {
            logits[bi * length + eos_token_id[i]] = -1e4;
        }
    }
}

template<>
__global__ inline void min_length_logits_process<half>(half* logits,
                                                       const int64_t *cur_len,
                                                       const int64_t *min_len,
                                                       const int64_t *eos_token_id,
                                                       const int64_t bs,
                                                       const int64_t length,
                                                       const int64_t end_length) {
    int bi = blockIdx.x;
    if (cur_len[bi] < 0) {
        return;
    }
    if (cur_len[bi] < min_len[bi]) {
        for (int i=0; i < end_length; i++) {
            logits[bi * length + eos_token_id[i]] = -1e4;
        }
    }
}


__global__ void update_repeat_times(const int64_t *pre_ids, 
                                    const int64_t *cur_len,
                                    int *repeat_times, 
                                    const int64_t bs, 
                                    const int64_t length, 
                                    const int64_t length_id) {
    int bi = blockIdx.x;
    if (cur_len[bi] < 0) {
        return;
    }
    int tid = threadIdx.x;
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    int *repeat_times_now = repeat_times + bi * length;
    for (int i = tid; i < length_id; i += blockDim.x) {
        int64_t id = pre_ids_now[i];
        if (id < 0) break;
        atomicAdd(&repeat_times_now[id], 1);
    }
}

template<typename T>
__global__ void update_value_by_repeat_times(const int *repeat_times, 
                                             const T *penalty_scores, 
                                             const T *frequency_score, 
                                             const T *presence_score, 
                                             T *logits, 
                                             const int64_t bs, 
                                             const int64_t length) {
    int bi = blockIdx.x;
    int tid = threadIdx.x;
    T *logits_now = logits + bi * length;
    const int *repeat_times_now = repeat_times + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = tid; i < length; i += blockDim.x) {
        int times = repeat_times_now[i];
        // printf("bi: %d, ti: %d, repeat_times: %d\n", bi, tid, times);
        if (times == 0) continue;
        float logit_now = static_cast<float>(logits_now[i]);
        logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
        logits_now[i] = static_cast<T>(logit_now - times * beta - gamma);
        // printf("bi: %d, ti: %d, repeat_times: %d, presence_score: %f, frequency_score: %f, presence_score: %f, logit_now: %f, logits: %f\n", 
                // bi, tid, times, alpha, beta, gamma, (float)logit_now, (float)logits_now[i]);
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> token_penalty_multi_scores_kernel(const paddle::Tensor& pre_ids, 
                                                              const paddle::Tensor& logits, 
                                                              const paddle::Tensor& penalty_scores, 
                                                              const paddle::Tensor& frequency_score, 
                                                              const paddle::Tensor& presence_score,
                                                              const paddle::Tensor& cur_len,
                                                              const paddle::Tensor& min_len,
                                                              const paddle::Tensor& eos_token_id) {

    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto cu_stream = logits.stream();
    std::vector<int64_t> shape = logits.shape();
    auto repeat_times = paddle::full(shape, 0, paddle::DataType::INT32, pre_ids.place());
    int64_t bs = shape[0];
    int64_t length = shape[1];
    int64_t length_id = pre_ids.shape()[1];
    auto logits_out = logits.copy_to(logits.place(), false); // gpu -> gpu

    int64_t end_length = eos_token_id.shape()[0];

    min_length_logits_process<<<bs, 1, 0, cu_stream>>>(
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits_out.data<data_t>())),
        cur_len.data<int64_t>(),
		min_len.data<int64_t>(),
		eos_token_id.data<int64_t>(),
		bs, length, end_length);

    int block_size_1 = (length_id + 32 - 1) / 32 * 32;
    block_size_1 = min(block_size_1, 512);
    update_repeat_times<<<bs, block_size_1, 0, cu_stream>>>(pre_ids.data<int64_t>(), cur_len.data<int64_t>(), repeat_times.data<int>(), bs, length, length_id);
    int block_size_2 = (length + 32 - 1) / 32 * 32;
    block_size_2 = min(block_size_2, 512);
    update_value_by_repeat_times<DataType_><<<bs, block_size_2, 0, cu_stream>>>(
        repeat_times.data<int>(),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(penalty_scores.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(frequency_score.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(presence_score.data<data_t>())),
        reinterpret_cast<DataType_*>(const_cast<data_t*>(logits_out.data<data_t>())),
        bs, length);
    return {logits_out};
}

std::vector<paddle::Tensor> TokenPenaltyMultiScores(const paddle::Tensor& pre_ids, 
                                                    const paddle::Tensor& logits, 
                                                    const paddle::Tensor& penalty_scores, 
                                                    const paddle::Tensor& frequency_scores, 
                                                    const paddle::Tensor& presence_scores,
                                                    const paddle::Tensor& cur_len,
                                                    const paddle::Tensor& min_len,
                                                    const paddle::Tensor& eos_token_id) {

    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            return token_penalty_multi_scores_kernel<paddle::DataType::BFLOAT16>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT16: {
            return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT16>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        case paddle::DataType::FLOAT32: {
            return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT32>(
                pre_ids,
                logits,
                penalty_scores,
                frequency_scores,
                presence_scores,
                cur_len,
                min_len,
                eos_token_id
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(const std::vector<int64_t>& pre_ids_shape, 
                                                                    const std::vector<int64_t>& logits_shape, 
                                                                    const std::vector<int64_t>& penalty_scores_shape, 
                                                                    const std::vector<int64_t>& frequency_scores_shape, 
                                                                    const std::vector<int64_t>& presence_scores_shape,
                                                                    const std::vector<int64_t>& cur_len_shape,
                                                                    const std::vector<int64_t>& min_len_shape,
                                                                    const std::vector<int64_t>& eos_token_id_shape) {
    return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(const paddle::DataType& pre_ids_dtype, 
                                                                const paddle::DataType& logits_dtype, 
                                                                const paddle::DataType& penalty_scores_dtype, 
                                                                const paddle::DataType& frequency_scores_dtype, 
                                                                const paddle::DataType& presence_scores_dtype,
                                                                const paddle::DataType& cur_len_dtype,
                                                                const paddle::DataType& min_len_dtype,
                                                                const paddle::DataType& eos_token_id_dtype) {
    return {logits_dtype};
}

PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"pre_ids", "logits", "penalty_scores", "frequency_scores", "presence_scores", "cur_len", "min_len", "eos_token_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));