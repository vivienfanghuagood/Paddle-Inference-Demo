from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=[
            'transfer_output.cc', 'save_with_output.cc', 'set_mask_value.cu', 'set_value_by_flags.cu', 'ngram_mask.cu', 'gather_idx.cu', 'token_penalty_multi_scores.cu', 
            'token_penalty_only_once.cu', 'stop_generation.cu', 'stop_generation_multi_ends.cu', 'set_flags.cu', 'fused_get_rope.cu']
    )
)

# setup(
#     name='custom_setup_ops',
#     ext_modules=CUDAExtension(
#         sources=[
#             'transfer_output.cc']
#     )
# )