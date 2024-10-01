# eval-ddp
# export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1

# DATASET=Video_Games
# MODEL=google-t5/t5-small
# OUTPUT_DIR=./ckpt/tiger/$DATASET/$MODEL-4x256-seqrec-item2index-step2
# # base_mode 4x256-seqrec-item2index/checkpoint-87000
# torchrun --nproc_per_node=2 --master_port=2309 finetune.py \
#     --base_model ./ckpt/tiger/Video_Games/google-t5/t5-small-4x256-seqrec-item2index-step2/checkpoint-31500 \
#     --resume_from_checkpoint ./ckpt/tiger/Video_Games/google-t5/t5-small-4x256-seqrec-item2index-step2/checkpoint-31500 \
#     --add_prefix_his \
#     --his_sep " " \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --per_device_batch_size 64 \
#     --learning_rate 1e-4 \
#     --epochs 20 \
#     --weight_decay 0.01 \
#     --save_and_eval_strategy steps \
#     --tasks seqrec \
#     --train_prompt_sample_num 1 \
#     --train_data_sample_num 0 \
#     --index_file .sentence_t5_4x256.index.json

# exp-2
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Video_Games
MODEL=google-t5/t5-small
OUTPUT_DIR=./ckpt/tiger/$DATASET/$MODEL-4x256-productsearch-item2index/

torchrun --nproc_per_node=2 --master_port=2309 finetune.py \
    --base_model $MODEL \
    --add_prefix_his \
    --his_sep " " \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 16 \
    --learning_rate 1e-3 \
    --epochs 20 \
    --weight_decay 0.01 \
    --save_and_eval_strategy steps \
    --tasks seqrec,productsearch,item2index \
    --train_prompt_sample_num 1,1,1 \
    --train_data_sample_num 0,0,0 \
    --index_file .sentence_t5_4x256.index.json


# export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0,1

# DATASET=Video_Games
# MODEL=google-t5/t5-small
# OUTPUT_DIR=./ckpt/tiger/$DATASET/TIGER-3x256-sentencet5-space-prefix/

# torchrun --nproc_per_node=2 --master_port=2309 finetune.py \
#     --resume_from_checkpoint ./ckpt/tiger/$DATASET/TIGER-3x256-sentencet5-space-prefix/checkpoint-34500 \
#     --add_prefix_his \
#     --his_sep " " \
#     --output_dir $OUTPUT_DIR \
#     --dataset $DATASET \
#     --per_device_batch_size 384 \
#     --learning_rate 1e-3 \
#     --epochs 100 \
#     --weight_decay 0.01 \
#     --save_and_eval_strategy steps \
#     --tasks seqrec \
#     --train_prompt_sample_num 1 \
#     --train_data_sample_num 0 \
#     --index_file .sentence_t5_3x256.index.json






